# Build from back/go:  docker build -t yair-controller:latest -f controller/Dockerfile .
FROM golang:1.25-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY controller ./controller
RUN CGO_ENABLED=0 go build -o /out/controller ./controller

FROM alpine:3.20
COPY --from=build /out/controller /controller
ENTRYPOINT ["/controller"]
