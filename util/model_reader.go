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
		count, err = mr.byteReader.Read(mr.buffer[numBytesRead:numBytes])
		if err != nil {
			panic(err)
		}
		if count < 0 {
			return numBytesRead, err
		}
	}

	return numBytesRead, nil
}

func (mr *ModelReader) ReadByteAsInt() (int, error) {
	b, err := mr.byteReader.ReadByte()
	return int(b), err
}

func (mr *ModelReader) ReadByteArray(numBytes int) ([]byte, error) {
	numBytesRead, err := mr.fillBuffer(numBytes)
	if err != nil {
		return nil, nil
	}
	if numBytesRead < numBytes {
		return nil, fmt.Errorf("Cannot read byte array (shortage): expected = %d, actual = %d", numBytes, numBytesRead)
	} else {
		result := make([]byte, numBytes)
		copy(mr.buffer[0:numBytes], result)
		return result, nil
	}
}

func (mr *ModelReader) ReadInt() (int, error) {
	return mr.ReadIntByteOrder(binary.LittleEndian)
}

func (mr *ModelReader) ReadIntBE() (int, error) {
	return mr.ReadIntByteOrder(binary.BigEndian)
}

func (mr *ModelReader) ReadIntByteOrder(order binary.ByteOrder) (int, error) {
	numBytesRead, err := mr.fillBuffer(4)
	if err != nil {
		return 0, err
	}
	if numBytesRead < 4 {
		return 0, fmt.Errorf("Cannot read int value (shortage): %d", numBytesRead)
	} else {
		return int(int32(order.Uint32(mr.buffer[0:4]))), nil
	}
}

func (mr *ModelReader) ReadIntArray(numValues int) ([]int, error) {
	numBytesRead, err := mr.fillBuffer(numValues * 4)
	if err != nil {
		return nil, err
	}
	if numBytesRead < numValues*4 {
		return nil, fmt.Errorf("Cannot read int array (shortage): expected = %d, actual = %d", numValues*4, numBytesRead)
	} else {
		res := make([]int, numValues)
		for i := 0; i < numValues; i++ {
			res[i] = int(int32(binary.LittleEndian.Uint32(mr.buffer[i*4:(i+1)*4])))
		}

		return res, nil
	}
}

func (mr *ModelReader) ReadUnsignedInt() (int, error) {
	result, err := mr.ReadInt()
	if err != nil {
		return 0, err
	}
	if (result < 0) {
		return 0, fmt.Errorf("Cannot read unsigned int (overflow): %d", result)
	} else {
		return result, nil
	}
}

func (mr *ModelReader) ReadInt64() (int64, error) {
	numBytesRead, err := mr.fillBuffer(8)
	if err != nil {
		return 0, err
	}
	if numBytesRead < 8 {
		return 0, fmt.Errorf("Cannot read long value (shortage): %d", numBytesRead)
	} else {
		return int64(binary.LittleEndian.Uint64(mr.buffer[0:8])), nil
	}
}

func (mr *ModelReader) AsFloat(bytes []byte) float32 {
	return float32(binary.LittleEndian.Uint32(mr.buffer[0:4]))
}

func (mr *ModelReader) AsUnsignedInt(bytes []byte) (int, error) {
	result := int(int32(binary.LittleEndian.Uint32(bytes)))
	if result < 0 {
		return 0, fmt.Errorf("Cannot treat as unsigned int (overflow): %d", result)
	} else {
		return result, nil
	}
}

func (mr *ModelReader) ReadFloat() (float32, error) {
	numBytesRead, err := mr.fillBuffer(4)
	if err != nil {
		return 0, err
	}
	if numBytesRead < 4 {
		return 0, fmt.Errorf("Cannot read float value (shortage): %d", numBytesRead)
	} else {
		return math.Float32frombits(binary.LittleEndian.Uint32(mr.buffer[0:4])), nil
	}
}

func (mr *ModelReader) ReadFloatArray(numValues int) ([]float32, error) {
	numBytesRead, err := mr.fillBuffer(numValues * 4)
	if err != nil {
		return nil, err
	}
	if numBytesRead < numValues*4 {
		return nil, fmt.Errorf("Cannot read float array (shortage): expected = %d, actual = %d", numValues*4, numBytesRead)
	} else {
		res := make([]float32, numValues)
		for i := 0; i < numValues; i++ {
			res[i] = math.Float32frombits(binary.LittleEndian.Uint32(mr.buffer[i*4:(i+1)*4]))
		}

		return res, nil
	}
}

func (mr *ModelReader) ReadDoubleArrayBE(numValues int) ([]float64, error) {
	numBytesRead, err := mr.fillBuffer(numValues * 8)
	if err != nil {
		return nil, err
	}
	if numBytesRead < numValues*8 {
		return nil, fmt.Errorf("Cannot read double array (shortage): expected = %d, actual = %d", numValues*8, numBytesRead)
	} else {
		res := make([]float64, numValues)
		for i := 0; i < numValues; i++ {
			res[i] = math.Float64frombits(binary.LittleEndian.Uint64(mr.buffer[i*8:(i+1)*8]))
		}

		return res, nil
	}
}

func (mr *ModelReader) Skip(numBytes int) error {
	numBytesRead, err := mr.byteReader.Discard(numBytes)
	if err != nil {
		return err
	}
	if numBytesRead < numBytes {
		return fmt.Errorf("Cannot skip bytes: %d", numBytesRead)
	}
	return nil
}

func (mr *ModelReader) ReadString() (string, error) {
	length, err := mr.ReadInt64()
	if err != nil {
		return "", err
	}
	if length > 2147483647 {
		return "", fmt.Errorf("Too long string: %d", length)
	} else {
		return mr.ReadFixedString(int(length))
	}
}

func (mr *ModelReader) ReadFixedString(numBytes int) (string, error) {
	numBytesRead, err := mr.fillBuffer(numBytes)
	if err != nil {
		return "", err
	}
	if numBytesRead < numBytes {
		return "", fmt.Errorf("Cannot read string(%d) (shortage): %d", numBytes, numBytesRead)
	} else {
		return string(mr.buffer[0:numBytes]), nil
	}
}
