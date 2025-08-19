## Tools

### Layers?

It seems like we're mixing metaphors to some extent - some things are layers, some things are not. for example it would appear that we can
construct a tool for an agent via Agent::with_tool (see e.g. examples/approval.rs), but we can ALSO construct a tool as a layer (e.g. typed_env_approval).
It's unclear if there's a difference or not. Perhaps this was done just to preserve the old interface?

### Typed interfaces

The typing of tools (e.g. examples/typed_tool.rs) is really slick. Love that.

### Tool context vs handlers vs tool functions

This is where I'm definitely feeling some confusion that's (I think) derived from the layers section above. We have context, which appears to be a way to
register something that should happen on tool outputs.

But that's separate and distinct from the actual tools themselves. And the tools could actually perform this themselves if they wanted - we could provide
a DI container during the creation of the tool if we wanted to. Not to suggest that the final DX we want is that, exactly, but it's _possible_ and makes it
unclear what value the context handler is adding.

To add even more muddying of the waters to the mix, this could ALSO be done with layers.

Not 100% sure what the right end DX is here, but it doesn't feel like this is it. I think it's more likely to be tools as services plus layers for
manipulating and/or retrieving data. typed_env_approval.rs feels the closest to me.

Tools should be doing the work of retrieving data/computing stuff, then we should be able to add layers to modify their behavior.
