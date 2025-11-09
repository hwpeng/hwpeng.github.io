---
title: The Weight of Scale
date: '2025-10-03'
---

I first read Rich Sutton’s famous [*The Bitter Lesson*](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) around 2021. As a struggling PhD student facing a string of paper rejections on AI accelerators, its message didn’t fully resonate with me at the time. My work in hardware seemed to be a constant game of catch-up, merely implementing the clever ideas of algorithm designers. I thought the real innovation was in the model, not the silicon.

Revisiting the essay now, with a few more years of experience under my belt, the lesson is starkly clear. It's 'bitter' because it's a humbling one for us as researchers: in the long run, our clever, human-designed solutions are consistently overtaken by simpler, more general methods that just leverage massive amounts of computation. The ultimate bottleneck, and therefore the ultimate driver of progress, isn't the elegance of an algorithm but the **raw power of the hardware** it runs on. The search for greater intelligence isn't an algorithm hunt, it's a hardware race.

This realization makes hardware the most exciting frontier in AI, a thrilling prospect for anyone in the field. But this relentless pursuit of performance also comes with a heavy burden -- **the weight of scale** -- a set of daunting challenges in programmability, sustainability, and fairness that we are now forced to confront.


## The Programmability Problem

With the end of easy transistor scaling, performance gains now come from designing chips and systems tailored for specific workloads, whether it’s AI, crypto, or something else. This specialization isn’t limited to the chip, it extends to programming languages, compilers, and system architecture. Each layer is optimized to squeeze out more performance, but this comes at the cost of flexibility and programmability. If a chip is 10x faster than an Nvidia GPU but nobody knows how to use it, it’s not much use in practice -- the famous CUDA moat.

The open-source ML community is actively working to improve programmability through projects like Triton and ROCm, making it easier for developers to write portable, high-performance code across different hardware platforms. These projects are promising, but the core problem isn't going away. Too often, we are building Ferraris of computation that handle like tractors. Closing the vast gap between the potential of our hardware and the practical ability of developers to harness it remains the first great hurdle in the race for scale.

## The Scaling Dilemma

Chip design is extremely challenging. From project kickoff to chip tape-out, the process can take two to three years and require the efforts of hundreds of people. If you also want a complete software stack and full system-level integration, that’s another two to three years and another large team, if not more. The only reason so many companies are still investing in AI chips is because the market is so massive that the NRE costs can be amortized to almost nothing compared to the potential returns ([ASIC Cloud](https://www.bsg.ai/papers/ASIC_Cloud_ISCA_2016_Proceedings.pdf), [Chiplet Cloud](https://arxiv.org/abs/2307.02666)). But even so, the returns from scaling up don’t always keep pace with the money and time invested. For any application, the performance-versus-investment curve eventually flattens out, and at some point, no one wants to keep playing that game.

Beyond financial costs, environmental impact is another major concern. The [carbon emissions](https://arxiv.org/abs/2104.10350) from training large neural networks are significant and growing. If new hardware truly makes people’s work more efficient, the trade-off might be justified, but at some point, the environmental cost will become too high to ignore. Designing more sustainable hardware systems is a big challenge, requiring a full stack commitment that includes designing hyper-efficient chip, using heterogeneous packaging with older process nodes, and building heterogeneous systems that can repurpose older hardware to extend its useful life. These approaches are essential if we want to keep advancing without running into insurmountable economic or environmental barriers.

## The Fairness Lottery
This might seem less relevant to hardware, but it’s something that has bothered me for a long time. Sarah Hooker’s concept of the [*hardware lottery*](https://arxiv.org/abs/2009.06489) describes how a research idea often succeeds not because it’s the best, but because it fits the available software and hardware. The success of DeepSeek proves this again: all they did was optimize their model to run efficiently on the GPUs they had. This highlights the importance of software-hardware co-design, but it also exposes a deeper, self-reinforcing cycle. As large companies invest vast resources in hardware for profitable applications, they create a deeply uneven playing field. The stark result is that not all research areas get equal opportunities to advance, and not everyone gets to benefit from these advancements. Ultimately, this dynamic allows the direction of scientific progress to be steered by a handful of companies whose commercial incentives often diverge from the ideal of equitable advancement.

The path forward, then, isn’t just about building faster hardware, but more accessible hardware. Making the chip design flow more open source, encouraging agile and modular hardware development, and embracing the trend of running AI on edge devices are all steps in the right direction. The fairness problem can sound cliché, but the stakes are real.  As the world becomes more divided, I hope that technology can at least avoid making things worse.

## *It was the best of times, it was the worst of times*

For every thrilling leap in performance, a new shadow is cast. We need hardware that doesn't just scale up for performance, but scales out – to more ideas and more people.

Alright, I think I've exceeded my context window for a Friday night. The real solution is to let the machine do the hard work anyway. To quote Sutton one last time:

> We want AI agents that can discover like we can, not which contain what we have discovered. Building in our discoveries only makes it harder to see how the discovering process can be done.


