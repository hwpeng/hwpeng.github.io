---
title: "Why I Will Never Buy Individual Stocks"
date: '2026-08-29'
originalLanguage: zh
---

A few years ago, I had been in the United States for more than five years. Finishing my PhD was still nowhere in sight, but for tax purposes I had become a resident, and I started thinking seriously about how to invest my savings.

Around the same time, I was studying reinforcement learning and happened upon the personal website of [Julian Schrittwieser](https://www.julian.ac/now/), one of the main developers of AlphaGo. He had [written down his philosophy of life](https://www.julian.ac/live/), and his investment advice was almost boringly simple: keep enough cash to live on for a while, regularly invest any money you will not need for ten years in index funds, and never buy individual stocks. He even made a set of [slides](https://docs.google.com/presentation/d/1kWsyBIKy42H20LpaGU-E6n0GJGRrpCTukzlv6VQRRdI/edit?slide=id.p#slide=id.p) to explain it.

That made perfect sense to me, so I set a few rough goals for my own investing:

- Control risk with as little effort as possible.
- Minimize taxes and fees.
- Only then, improve the return on spare cash: first beat inflation, then earn the average market return, with no ambition to outperform it.

That became my investment strategy: put spare money into index funds on a regular schedule. After I started working, I also sold all of my company RSUs as soon as they vested instead of holding on to them.

But technology stocks have risen so much over the past few years. I know plenty of people who made a great deal of money on NVIDIA and Micron, while relatives and friends in China have always traded A-shares. Naturally, I sometimes ask myself: should I spend a little time researching individual companies?

I work on AI chips, and I enjoy studying industry trends. I would confidently say that I understand demand for AI infrastructure, semiconductors, and HBM better than the vast majority of retail investors. If other people have made money on these stocks, why can't I turn my professional knowledge into investment returns too?

After thinking about it seriously, my answer is still the same: I will never buy individual stocks. In fact, for most people, I do not think it is worth doing at all.

## Everything Is Priced In

My old understanding of the stock market was simple: it contains enormous noise and very little signal, so collecting information and using it to predict trends is extraordinarily complicated and exhausting. That was not wrong, but it was not deep enough. It still suggested that with enough information and enough expertise, one could beat the market.

What truly made me reconsider was the enormous difference between the stock performance of NVIDIA and TSMC.

Over the past few years, NVIDIA has risen far more than TSMC. Yet if I look only at the supply chain and technical barriers, I find this hard to explain. AI requires NVIDIA GPUs, but at least AMD exists, and more companies are designing their own accelerators. No matter who designs the chip, most advanced products still need to be manufactured by TSMC. NVIDIA itself cannot do without TSMC.

If I had to choose which of the two companies was harder to replace, I might actually choose TSMC. So why did NVIDIA rise more?

I eventually realized that the key question is not whether TSMC is a good company. It is when everyone learned that it was a good company.

Intel did not fall behind in advanced process technology only after ChatGPT appeared. GlobalFoundries had already put its 7 nm program on indefinite hold and withdrawn from the leading-edge race in [2018](https://investors.gf.com/news-releases/news-release-details/globalfoundries-reshapes-technology-portfolio-intensify-focus). Apple had long entrusted its most advanced A-series chips to TSMC. Well before GenAI became an investment theme, the market knew that TSMC had leading technology, excellent customers, and almost no alternative at the most advanced nodes.

ChatGPT did not suddenly make TSMC a good company. It had always been one, and everyone had known for a long time.

That was when I finally understood what priced-in really means.

Stock returns do not depend on whether a company will be good in the future. They depend on how much better its future will be than the expectations already embedded in today's price. "Advanced chips cannot exist without TSMC" explains why TSMC is valuable. It does not explain why the stock should outperform the market starting today. If those advantages are already part of the consensus, they are already in the price.

I do not merely need to predict the future. I need to know what future everyone else has already priced in, and then predict how reality will differ from that consensus.

ASML takes the problem one level further. It has a near monopoly on EUV lithography systems, with a technical moat that may be even more extreme than TSMC's. The simplest supply-chain analysis goes like this: AI needs advanced chips; advanced chips need EUV; only ASML can build EUV systems; therefore ASML should be one of the largest beneficiaries. Yet during the first few years of the GenAI boom, its stock made almost no progress from its 2021 high through the end of 2024, dramatically underperforming NVIDIA.

ASML's monopoly was also public knowledge. But there is another problem here: being the technical bottleneck does not mean being the part of the chain where profits grow fastest. Between AI demand and ASML revenue lie accelerator demand, advanced-node wafer demand, fab expansion, and lithography intensity.

NVIDIA benefits directly from GPU demand, pricing, and product mix. ASML sells capital equipment that can operate for many years. Selling ten times as many GPUs does not mean fabs will immediately buy ten times as many EUV systems. Customers can first increase utilization, then upgrade existing machines, and only later build new capacity. "Nothing works without it" and "it captures most of the incremental profit" have never meant the same thing.

More interestingly, ASML has surged again this year. In the second quarter of 2026, the company [raised its full-year revenue outlook](https://www.asml.com/en/news/press-releases/2026/q2-2026-financial-results) because AI demand had finally begun to translate into more aggressive capacity plans from its customers. ASML did not suddenly acquire its EUV monopoly; it had held that monopoly all along. What changed was the market's estimate of how quickly the monopoly could turn into orders.

Micron presents yet another problem. I am quite confident that HBM demand will continue to grow. Then what? Does that tell me Micron's stock price next year?

Memory has always been a highly cyclical business. Higher demand brings higher prices and profits, but it also leads manufacturers to expand capacity, customers to adjust purchasing, and more investors to buy in ahead of time. Even if I predict future HBM bit demand perfectly, I still need to know what the market has already priced in, how supply will respond, how long prices can hold, and when everyone else will change their minds.

This is not a stationary distribution. When all the participants are observing and making decisions, the object being observed changes with them. Once a profitable pattern is discovered, it attracts more capital until it stops working. We try to learn the market from history, while the market is also learning us.

Professional knowledge can help me understand an industry and do my job well. It cannot tell me how much of that understanding has already been absorbed into the price.

Only a small minority of funds beat the market over long periods, while the returns from individual stocks are concentrated in a tiny number of super-winners. An index does not need to know the winners in advance. Stock picking does. The problem is that before they become stories, I do not know who they are.

## Every Winning Bet Looks Like Skill

If stock picking is so difficult, why do so many people remain obsessed with it? Greed is certainly one answer, but I think there is a deeper reason: once both money and self-worth are involved, people find it very difficult to keep thinking rationally for long.

When people make money on a stock, they naturally believe that their judgment was correct. Even if they earn less than they expected, they merely regret not investing more, perhaps even not using leverage. But when they lose money, they rarely question the method. Instead, they blame themselves for failing to overcome greed or fear, or for not cutting their losses quickly enough.

That is why phrases like "be fearful when others are greedy and greedy when others are fearful" or "the stock market goes against human nature" all sound like bullshit to me. If you make money, the method was right. If you lose money, you simply have not mastered yourself yet.

It is a system that cannot be falsified.

For most people, it is no different from gambling. Not because there is no real edge in the market, but because too many random outcomes are mistaken for skill. A gambler rolls several sixes in a row and starts believing that he has found the correct wrist motion—that he has become the God of Gamblers.

Stock picking also provides more than returns. It offers a powerful form of intellectual validation: I understood this before everyone else. When I make money while others lose it, the pleasure may be even stronger, because it seems to prove not only that I have money, but that I am smarter. Index investing offers none of this satisfaction. When the index rises, no one can take personal credit.

Leopold's story is an extreme example. He is exceptionally intelligent, and many of his long-term judgments about AI infrastructure may ultimately prove correct. His fund, Situational Awareness, once produced astonishing returns. Then it fell 67% in a single month this July, forcing him to sell most of the public-equity portfolio to Citadel and unwind the leverage.

The story does not prove that Leopold's industry judgment was wrong. It shows something else: enormous success brings more capital, greater conviction, and permission to take greater risks. Once position size, leverage, and liquidity are all magnified, being "right in the long run" may no longer be enough. The market does not have to prove that you are wrong forever. It only has to make sure you cannot survive until the day you are right.

I firmly believe that I am smarter than most people at many things. I also know that this very conviction is exactly the part of human nature I should distrust most. I do not need a stock price to prove the first sentence, and I cannot afford to forget the second.

## Not Every Profit Is Worth Earning

My understanding of investing and economic development is rather simple: in the long run, economic growth ultimately comes from higher productivity. Society becomes genuinely wealthier when we invent better technology, produce more with fewer resources, and allow more people to live better lives.

I admire technological progress, which is also why I have always found value investing appealing. I want my returns to come from companies genuinely creating new value.

But from what I have observed, that is not how most people talk about trading stocks. Of course they say that they believe in a company's long-term value. Yet as soon as the conversation turns to actual investing technique, the questions always become: what has the market not discovered yet? Can I buy before everyone else? When will other people follow? How can I sell before they do?

To put it bluntly: how do I make sure I am not the last fool holding the bag?

At that point, returns no longer come only from how much value a company creates. They also depend on whether I can correctly guess that the next person will pay a higher price. My excess return requires someone else to discover the information later, make a worse judgment, or absorb the final loss. That does not fit my moral intuition. Even if I really am smarter than other people, it does not mean I want to use that intelligence to profit from their losses.

I have no intention of passing moral judgment on anyone. My unwillingness to make money this way does not mean that I deny its usefulness to the market. Those are two different questions.

Economic development requires not only that capital eventually reach productive uses, but that it get there quickly. New information needs to be reflected in prices as soon as possible. People who want to exit need to be able to sell at any time. Capital needs to move from companies with declining prospects toward more valuable ones.

Speculators do exactly this. In their effort to buy and sell before everyone else, they constantly search for new information and put their judgments into the price. Their trades provide liquidity and help the market discover prices faster.

But the market's need for speculation does not mean that more trading is always better. Kenneth French estimated that between 1980 and 2006, American investors spent an average of 0.67% of the total value of the market each year on fees and trading costs in pursuit of excess returns. He described this as the social cost paid for price discovery; for most investors, moving to passive investing would actually improve returns. [The Cost of Active Investing](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1105775)

Even if I someday discover an absolutely reliable edge, if it requires me to understand information earlier than other people and profit from their mistakes, I still do not want it. The market already has enough people searching for such opportunities. It will not miss me.

## What Are You Optimizing For?

Before making any investment, perhaps we should ask a more basic question: what exactly am I doing this for?

If the answer is more money, what do I want that money for, and how do I plan to use it? If the point is not money, but the intellectual pleasure of research or the excitement of prices moving up and down, that is fine too. It is simply better to be honest with yourself instead of mistaking an expensive form of entertainment for a rational financial plan.

I once had an older roommate. He bought a substantial amount of Tesla stock very early and, for a while, made a great deal of money. He asked whether I wanted to buy some too. I did not. Later, he sold at a very good price and achieved the sort of investment outcome many people dream of.

But the story did not end on the day he sold. Later, he fell victim to an online romance scam and lost nearly all of his savings. The following year, he could not even afford the taxes owed on the stock sale and had to take out loans.

This was not his fault, nor does it mean that someone deserves to be scammed if they have not decided how to use their money. It simply made me realize that acquiring wealth, protecting wealth, and using wealth are three entirely different abilities. When we talk about investing, we often direct all our attention toward the first, as if the story has already reached a happy ending once the number in the account becomes large enough.

We constantly discuss the ROI of an investment, but rarely begin by deciding what R and I actually mean.

If the R is only a number in an account, the number itself cannot make life better. Money matters only when it buys security, freedom, and a better real life. If the I includes only the principal, it misses more important costs: the time and attention spent researching stocks, and everything else I gave up as a result. Money can be earned again. Lost time cannot.

Choosing an investment, then, cannot be reduced to comparing which one has the higher rate of return. Only after understanding what you truly want, and what you are willing to pay for it, can you make a choice suited to your own life.

For me, instead of researching which stock will rise next quarter, I would rather spend the time improving at my work or being with my family. Of course, I should still thank everyone who keeps buying technology stocks. Their enthusiasm for AI has brought more capital into the field and helped me land a job that pays reasonably well.

Maybe trading stocks really could make me more money. Maybe it could not. It no longer matters. Attention Is All You Need, and my attention belongs elsewhere.
