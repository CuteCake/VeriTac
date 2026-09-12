# Build the bounded ANE feasibility probe (adapted from maderix/ANE, MIT).
# See LICENSE-ANE.md and README.md.

CC      := xcrun clang
CFLAGS  := -O2 -Wall -Wextra -Wno-deprecated-declarations -fobjc-arc
FRAMEWORKS := -framework Foundation -framework IOSurface -ldl

SRC := src/ane_bridge.m src/ane_probe.m
TARGET := ane_probe

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) src/ane_bridge.h
	$(CC) $(CFLAGS) -o $@ $(SRC) $(FRAMEWORKS)

clean:
	rm -f $(TARGET)
