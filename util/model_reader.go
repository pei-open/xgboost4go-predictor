package util

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

type ModelReader struct {
	buffer     []byte
	byteReader bufio.Reader
}

func NewModelReaderByFile(fileName string) (*ModelReader, *os.File, error) {
	modelReader := new(ModelReader)
	file, err := os.Open(fileName)
	if err != nil {
		return modelReader, file, err
	}
	return NewModelReaderByReader(*bufio.NewReader(file)), file, err
}

func NewModelReaderByReader(reader bufio.Reader) *ModelReader {
	modelReader := new(ModelReader)
	modelReader.byteReader = reader
	return modelReader
}

func (mr *ModelReader) fillBuffer(numBytes int) (int, error) {
	if mr.buffer == nil || len(mr.buffer) < numBytes {
		mr.buffer = make([]byte, numBytes)
	}

	count := 0
	numBytesRead := 0
	var err error
	for ; numBytesRead < numBytes; numBytesRead += count {
		count, err = mr.byteReader.Read(mr.buffer[numBytesRead:numBytes-numBytesRead])
		if (count < 0) {
			return numBytesRead, err
		}
	}

	return numBytesRead, nil
}

func (mr *ModelReader) ReadByteAsInt() (int, error) {
	b, err := mr.byteReader.ReadByte()
	return int(b), err
}

func (this *ModelReader) ReadByteArray(numBytes int) ([]byte, error) {
	numBytesRead, err := this.fillBuffer(numBytes)
	if err != nil {
		return nil, nil
	}
	if numBytesRead < numBytes {
		return nil, fmt.Errorf("Cannot read byte array (shortage): expected = %d, actual = %d", numBytes, numBytesRead)
	} else {
		result := make([]byte, numBytes)
		copy(this.buffer[0:numBytes], result)
		return result, nil
	}
}

func (mr *ModelReader) ReadInt() (int, error) {
	return mr.ReadIntByteOrder(binary.LittleEndian)
}

func (mr *ModelReader) ReadIntBE() (int, error) {
	return mr.ReadIntByteOrder(binary.BigEndian)
}

func (this *ModelReader) ReadIntByteOrder(order binary.ByteOrder) (int, error) {
	numBytesRead, err := this.fillBuffer(4)
	if err != nil {
		return 0, err
	}
	if numBytesRead < 4 {
		return 0, fmt.Errorf("Cannot read int value (shortage): %d", numBytesRead)
	} else {
		return int(order.Uint32(this.buffer[0:4])), nil
	}
}

func (this *ModelReader) ReadIntArray(numValues int) ([]int, error) {
	numBytesRead, err := this.fillBuffer(numValues * 4)
	if err != nil {
		return nil, err
	}
	if numBytesRead < numValues*4 {
		return nil, fmt.Errorf("Cannot read int array (shortage): expected = %d, actual = %d", numValues*4, numBytesRead)
	} else {
		res := make([]int, numValues)
		for i := 0; i < numValues; i++ {
			res[i] = int(binary.LittleEndian.Uint32(this.buffer[i*4:(i+1)*4]))
		}

		return res, nil
	}
}

func (this *ModelReader) ReadUnsignedInt() (int, error) {
	result, err := this.ReadInt()
	if err != nil {
		return 0, err
	}
	if (result < 0) {
		return 0, fmt.Errorf("Cannot read unsigned int (overflow): %d", result)
	} else {
		return result, nil
	}
}

func (this *ModelReader) ReadInt64() (int64, error) {
	numBytesRead, err := this.fillBuffer(8)
	if err != nil {
		return 0, err
	}
	if (numBytesRead < 8) {
		return 0, fmt.Errorf("Cannot read long value (shortage): %d", numBytesRead)
	} else {
		return int64(binary.LittleEndian.Uint64(this.buffer[0:8])), nil
	}
}

func (this *ModelReader) AsFloat(bytes []byte) float32 {
	return float32(binary.LittleEndian.Uint32(this.buffer[0:4]))
}

func (this *ModelReader) AsUnsignedInt(bytes []byte) (int, error) {
	result := int(binary.LittleEndian.Uint32(bytes))
	if (result < 0) {
		return 0, fmt.Errorf("Cannot treat as unsigned int (overflow): %d", result)
	} else {
		return result, nil
	}
}

func (this *ModelReader) ReadFloat() (float32, error) {
	numBytesRead, err := this.fillBuffer(4)
	if err != nil {
		return 0, err
	}
	if (numBytesRead < 4) {
		return 0, fmt.Errorf("Cannot read float value (shortage): %d", numBytesRead)
	} else {
		return math.Float32frombits(binary.LittleEndian.Uint32(this.buffer[0:4])), nil
	}
}

func (this *ModelReader) ReadFloatArray(numValues int) ([]float32, error) {
	numBytesRead, err := this.fillBuffer(numValues * 4)
	if err != nil {
		return nil, err
	}
	if (numBytesRead < numValues*4) {
		return nil, fmt.Errorf("Cannot read float array (shortage): expected = %d, actual = %d", numValues*4, numBytesRead)
	} else {
		res := make([]float32, numValues)
		for i := 0; i < numValues; i++ {
			res[i] = math.Float32frombits(binary.LittleEndian.Uint32(this.buffer[i*4:(i+1)*4]))
		}

		return res, nil
	}
}

func (this *ModelReader) ReadDoubleArrayBE(numValues int) ([]float64, error) {
	numBytesRead, err := this.fillBuffer(numValues * 8)
	if err != nil {
		return nil, err
	}
	if (numBytesRead < numValues*8) {
		return nil, fmt.Errorf("Cannot read double array (shortage): expected = %d, actual = %d", numValues*8, numBytesRead)
	} else {
		res := make([]float64, numValues)
		for i := 0; i < numValues; i++ {
			res[i] = math.Float64frombits(binary.LittleEndian.Uint64(this.buffer[i*8:(i+1)*8]))
		}

		return res, nil
	}
}

func (this *ModelReader) Skip(numBytes int) error {
	numBytesRead, err := this.byteReader.Discard(numBytes)
	if err != nil {
		return err
	}
	if (numBytesRead < numBytes) {
		return fmt.Errorf("Cannot skip bytes: %d", numBytesRead)
	}
	return nil
}

func (this *ModelReader) ReadString() (string, error) {
	length, err := this.ReadInt64()
	if err != nil {
		return "", err
	}
	if length > 2147483647 {
		return "", fmt.Errorf("Too long string: %d", length)
	} else {
		return this.ReadFixedString(int(length))
	}
}

func (this *ModelReader) ReadFixedString(numBytes int) (string, error) {
	numBytesRead, err := this.fillBuffer(numBytes)
	if err != nil {
		return "", err
	}
	if (numBytesRead < numBytes) {
		return "", fmt.Errorf("Cannot read string(%d) (shortage): %d", numBytes, numBytesRead)
	} else {
		return string(this.buffer[0:numBytes]), nil
	}
}
