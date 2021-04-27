# Virtual Interviewer: Build your own interviewing agent 🤖
This repo contains all the resources required for the AMLD 2021 workshop - Virtual Interviewer: Build your own interviewing agent.

The goal of this workshop is to build a virtual interviewing agent capable of asking dynamic follow-up questions. 

This workshop will be conducted in two parts. In the first part, we will see how to build a follow-up question generation model using pre-trained GPT-2 language model. In the second part, we will see a demo of deploying the built model as a virtual agent using AWS Sumerian.

## Part 1
Follow along with the workshop using the colab notebook

[![](https://colab.research.google.com/assets/colab-badge.svg)](https://tinyurl.com/yfunb7t2)

## Part 2
This will be a walk through of setting things up on AWS Sumerian to create a virtual agent with capabilities of interviewing and follow-up questioning. We will use the model trained in the Part 1.

:arrow_forward: [Read more](part-2/) 


## Slides
Find the slides for the workshop [here](https://tinyurl.com/y6d3cptx).

## Reference
This repository borrows code from [transfer-learning-conv-ai](https://github.com/huggingface/transfer-learning-conv-ai) of [HuggingFace](https://huggingface.co/).

## Cite
If you use this in your research please consider citing
```bash
@article{sb2021improving,
  title={Improving Asynchronous Interview Interaction with Follow-up Question Generation.},
  author={S. B, Pooja Rao and Agnihotri, Manish and Jayagopi, Dinesh Babu},
  journal={International Journal of Interactive Multimedia \& Artificial Intelligence},
  volume={6},
  number={5},
  year={2021}
}
```