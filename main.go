package main

import (
	"github.com/argylelabcoat/mempalace-go/cmd/cli"
	"github.com/argylelabcoat/mempalace-go/cmd/server"
)

func main() {
	root := cli.NewCommand()
	root.AddCommand(server.NewCommand())
	root.Execute()
}
