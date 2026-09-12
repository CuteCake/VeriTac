# Read-only ANE runtime capability inventory probe (see README.md).
CC      := xcrun clang
CFLAGS  := -O2 -Wall -Wextra -fobjc-arc
FRAMEWORKS := -framework Foundation -ldl
SRC     := src/ane_inventory.m
TARGET  := ane_inventory
OUT     := results/inventory.json

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(FRAMEWORKS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
