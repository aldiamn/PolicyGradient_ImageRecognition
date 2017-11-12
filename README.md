# Policy Gradient for MNIST Image Recognition
## Introduction
This project is a simple implementation for optimizing the neural network with policy gradient.  I built a mnist environment which provides the observation (images), and an agent (neural network). If the agent give the correct answer, the environment gives +1 as reward. However, the environment gives -1 as punishment. The agent will learn the information base on the reward it get. I take the gumbel softmax to do the exploration. The temperture will decay in the training procedure.

## Result
<figure class="half">
    <img src='fig/20171112-1.PNG'>
    <img src='fig/20171112-2.PNG'>
</figure>
<figure class="half">
    <img src='fig/20171112-3.PNG'>
    <img src='fig/20171112-4.PNG'>
</figure>