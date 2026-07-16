
warmup problem: 200 or 10% steps for all?


running exps:

lcfx: hotpotqa, qasc
eiyw: ropes, qasc
t2t:  

We need to redesign a new archietecture, which we call xkv. It will use KV from both sharer and receiver, and generate KV update for receiver. The two KV caches will be processed by the lcfx polling process, then concatenated, bottlenecked, and processed by a translator with will use attention machenism, finally the output part generate KV cache updates like in lcfx. Write a plan for this method and draw a figure to show its archietecture
