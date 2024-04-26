package main

import (
	"flag"
	"log"
	"os"
	"time"

	llama2 "github.com/mengzhuo/llama2.go"
	"github.com/mengzhuo/nn"
)

func main() {

	var (
		checkpointFilePath string
		tokenizerFilePath  string
		temperature        float64
		steps              int
		prompt             string
		topp               float64
	)

	flag.StringVar(&checkpointFilePath, "checkpoint", "out/model.bin", "checkpoint binary file with weights")
	flag.StringVar(&tokenizerFilePath, "tokenizer", "tokenizer.bin", "tokenizer binary file with vocabulary (get it from repo)")
	flag.Float64Var(&temperature, "temperature", 0.9, "temperature (optional; 0 = deterministic argmax sampling; 1 = baseline)")
	flag.IntVar(&steps, "steps", 256, "max number of steps to run for, 0: use seq_len")
	flag.Float64Var(&topp, "topp", 0.9, "top-p in nucleus sampling (1.0 = off; 0.9 works well, but slower)")
	flag.StringVar(&prompt, "prompt", "", "query to start with")
	flag.Parse()

	checkpointFile, err := os.Open(checkpointFilePath)
	if err != nil {
		log.Fatal(checkpointFilePath, err)
	}
	defer checkpointFile.Close()

	out := os.Stdout

	config, err := llama2.NewConfig(checkpointFile)
	if err != nil {
		log.Fatalf("cannot read config: %s", err)
	}
	log.Printf("config: %#v\n", config)

	// "negative vocab size is hacky way of signaling unsahred weights. biy yikes"  @karpathy
	isSharedWeights := config.VocabSize > 0
	if config.VocabSize < 0 {
		config.VocabSize = -config.VocabSize
	}

	tokenizerFile, err := os.Open(tokenizerFilePath)
	if err != nil {
		log.Fatal(tokenizerFilePath, err)
	}
	defer tokenizerFile.Close()

	vocab := llama2.NewVocabFromFile(config.VocabSize, tokenizerFile)

	w := llama2.NewTransformerWeightsFromCheckpoint(config, checkpointFile, isSharedWeights)

	// right now we cannot run for more than config.SeqLen steps
	if steps <= 0 || int32(steps) > config.SeqLen {
		steps = int(config.SeqLen)
	}

	runState := llama2.NewRunState(config)

	promptTokens := vocab.Encode(prompt)

	// the current position we are in
	timeStart := time.Now()
	var token int32 = 1 // 1 = BOS token in llama-2 sentencepiece
	var pos int32 = 0
	for pos < int32(steps) {
		// forward the transformer to get logits for the next token
		llama2.Transformer(token, pos, *config, runState, w)

		var next int
		if pos < int32(len(promptTokens)) {
			next = promptTokens[pos]
		} else {
			// sample the next token
			if temperature == 0 {
				// greedy argmax sampling
				next = nn.ArgMax(runState.Logits)
			} else {
				// apply the temperature to the logits
				for q := int32(0); q < config.VocabSize; q++ {
					runState.Logits[q] /= float32(temperature)
				}
				// apply softmax to the logits to the probabilities for next token
				nn.SoftMax(runState.Logits)
				// we now want to sample from this distribution to get the next token
				if topp <= 0 || topp >= 1 {
					// simply sample from the predicted probability distribution
					next = nn.Sample(runState.Logits)
				} else {
					// top-p (nucleus) sampling, clamping the least likely tokens to zero
					next = nn.SampleTopP(runState.Logits, float32(topp))
				}
			}
		}
		pos++

		// data-dependent terminating condition: the BOS (1) token delimits sequences
		if next == 1 {
			break
		}

		// following BOS (1) token, sentencepiece decoder strips any leading whitespace
		var tokenStr string
		if token == 1 && vocab.Words[next][0] == ' ' {
			tokenStr = vocab.Words[next][1:]
		} else {
			tokenStr = vocab.Words[next]
		}
		out.WriteString(tokenStr)

		// advance forward
		token = int32(next)
	}
	out.Write([]byte("\n"))

	log.Printf("achieved tok/s: %f\n", float64(pos-1)/time.Since(timeStart).Seconds())
}
