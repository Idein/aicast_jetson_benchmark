CC = gcc
CFLAGS = -W -Wall -Wextra -Werror -O2 -pipe -fPIC -std=c99
LDFLAGS = -shared
LIBS = -lhailort -lpthread -lm

SRC_DIR = src
SRC = $(SRC_DIR)/model.c
OBJ = $(SRC:.c=.o)

TARGET = yolox_s.so

ARM_FLAGS = -mcpu=arm1176jzf-s -mfpu=vfp -mfloat-abi=hard

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) $(ARM_FLAGS) -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) $(ARM_FLAGS) -I$(SRC_DIR) -c $< -o $@

clean:
	rm -f $(SRC_DIR)/*.o $(TARGET)

.PHONY: all clean
