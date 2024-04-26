package llama2

import (
	"encoding/binary"
	"io"
)

// NewTransformerWeightsFromCheckpoint reads binary checkpoint into weights.
// Notes on llama2.c: for checkpoint not using `mmap`, instead scanning file
func NewTransformerWeightsFromCheckpoint(config *Config, r io.Reader, isSharedWeights bool) TransformerWeights {
	w := TransformerWeights{
		TokenEmbeddingTable: make([]float32, (config.VocabSize * config.Dim)),
		RMSAttentionWeight:  make([]float32, (config.NumLayers * config.Dim)),
		RMSFFNWeight:        make([]float32, (config.NumLayers * config.Dim)),
		RMSFinalWeight:      make([]float32, config.Dim),
		WQ:                  make([]float32, (config.NumLayers * config.Dim * config.NumHeads * config.HeadSize())),
		WK:                  make([]float32, (config.NumLayers * config.Dim * config.NumKVHeads * config.HeadSize())),
		WV:                  make([]float32, (config.NumLayers * config.Dim * config.NumKVHeads * config.HeadSize())),
		WO:                  make([]float32, (config.NumLayers * config.NumHeads * config.HeadSize() * config.Dim)),
		W1:                  make([]float32, (config.NumLayers * config.Dim * config.HiddenDim)),
		W2:                  make([]float32, (config.NumLayers * config.HiddenDim * config.Dim)),
		W3:                  make([]float32, (config.NumLayers * config.Dim * config.HiddenDim)),
		FreqCISReal:         make([]float32, (config.SeqLen * config.HeadSize() / 2)),
		FreqCISImag:         make([]float32, (config.SeqLen * config.HeadSize() / 2)),
		WCLS:                make([]float32, (config.VocabSize * config.Dim)),
	}

	binary.Read(r, binary.LittleEndian, w.TokenEmbeddingTable)
	binary.Read(r, binary.LittleEndian, w.RMSAttentionWeight)
	binary.Read(r, binary.LittleEndian, w.WQ)
	binary.Read(r, binary.LittleEndian, w.WK)
	binary.Read(r, binary.LittleEndian, w.WV)
	binary.Read(r, binary.LittleEndian, w.WO)
	binary.Read(r, binary.LittleEndian, w.RMSFFNWeight)
	binary.Read(r, binary.LittleEndian, w.W1)
	binary.Read(r, binary.LittleEndian, w.W2)
	binary.Read(r, binary.LittleEndian, w.W3)
	binary.Read(r, binary.LittleEndian, w.RMSFinalWeight)
	binary.Read(r, binary.LittleEndian, w.FreqCISReal)
	binary.Read(r, binary.LittleEndian, w.FreqCISImag)

	if isSharedWeights {
		w.WCLS = w.TokenEmbeddingTable
	} else {
		binary.Read(r, binary.LittleEndian, w.WCLS)
	}

	return w
}
