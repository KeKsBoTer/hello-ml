
package = github.com/KeKsBoTer/hello-ml

all: clean build run

build:
	go build -o build $(package)

release:
	env GOOS=linux GOARCH=amd64  go build -ldflags "-s -w" -o gookie $(package)

run:
	./build 

clean:
	rm -f ./build