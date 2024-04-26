package llama2

import (
	"encoding/binary"
	"io"
)

type Config struct {
	Dim        int32 // transformer dimension
	HiddenDim  int32 // for FFN layers
	NumLayers  int32
	NumHeads   int32 // number of query heads
	NumKVHeads int32 // number of key/value heads (can be < query heads because of multiquery)
	VocabSize  int32 // usually 256 (byte level)
	SeqLen     int32 // max sequence length
}

func NewConfig(rd io.Reader) (*Config, error) {
	cfg := &Config{}
	err := binary.Read(rd, binary.LittleEndian, cfg)
	return cfg, err
}

func (c *Config) HeadSize() int32 { return c.Dim / c.NumHeads }

func (c *Config) KVDim() int32 { return (c.Dim * c.NumKVHeads) / c.NumHeads }

// KVMul integer multiplier of the kv sharing in multiquery
func (c *Config) KVMul() int32 { return c.NumHeads / c.NumKVHeads }
