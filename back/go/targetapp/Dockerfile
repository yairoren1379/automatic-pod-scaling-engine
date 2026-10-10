# Build from back/go:  docker build -t yair-api:latest -f targetapp/Dockerfile .
FROM golang:1.25-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
COPY targetapp ./targetapp
RUN CGO_ENABLED=0 go build -o /out/targetapp ./targetapp

FROM alpine:3.20
COPY --from=build /out/targetapp /targetapp
EXPOSE 8080
ENTRYPOINT ["/targetapp"]
