# Build from back/go:  docker build -t yair-loadgen:latest -f loadgen/Dockerfile .
FROM golang:1.25-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
COPY loadgen ./loadgen
RUN CGO_ENABLED=0 go build -o /out/loadgen ./loadgen

FROM alpine:3.20
COPY --from=build /out/loadgen /loadgen
ENTRYPOINT ["/loadgen"]
