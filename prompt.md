
# Echo In Your Words
This project aims to optimize the agent-to-agent communications by sharing the sharer agent's context in the format of receiver's tokens.

# Background

Agent to agent communications usually use text. But generating and encoding the texts are unnacessary if no human would read them. Therefore, we may let the sharer sends its internal state, e.g., KV cache (after processing its context), to the receiver, after a translator neural network that aligns the two models' token/latent space. 

The general workflow will be:
1. Sharer agent processes its context,
2. translator model handles the state of the sharer and output processed context to receiver,
3. receiver accepts the output from translator and finish generation

For step 2, the translator may handle the sharer in various ways, like compress its KV cache, or last layers, and the output can also have many options, such as KV cache update for the receiver.

The novelty of this project is that we can compress the sharer's context (its KV cache) to vectors attached to the receiver's tokens. That is, the translator will translate the context from the sharer into vectors somehow aligns with receiver's tokenizer, because those vectors are esentially tokens of the second model.

Existing works include:

Paper 1: Cache-to-cache: Direct semantic communication between large language models.

Paper 2: Latent Cache Flow: Model-to-Model Communication Without Text

# Tasks

1. We need to reproduce the results from the above two papers as 