## Poster Generator
Chloe, David, Nate, Carson, and Pei Pei

## Introduction Draft
Our goal is to generate movie poster images from random noise that are as close to human created posters as possible. This problem is difficult to solve because movie posters tend to be more nuanced compared to other kinds of images. Our proposed solution to this problem is to divide our dataset into movie posters separated by genre with the goal of reducing the amount of variance across the art/photo styles of the posters, which hopefully would make it easier for the model to learn to generate them.
    
This problem is difficult to solve because movie posters tend to be more nuanced compared to other kinds of images. Firstly, because movie posters are designed specifically to market the movie they depict to an audience, they tend to have many intentional and minute details which a model would struggle to learn. Secondly, movie posters tend to have cohesive designs which highlight several different important objects. In comparison, most other images tend to highlight a specific subject with less attention given to other details. So, aside from the subject, the rest of the image is closer to random noise, something that is easier for a model to learn compared to that of a movie poster. Recent generative art has struggled with emphasizing clearly recognizable elements, instead focusing on emulating stylistic aspects of different genres. 
    
During our exploration of the data, we realized that there is a wide variety of art and photography styles used across the 41,000 movie posters that would likely conflict enough with each other to make it hard for the model to learn to generate  posters with a distinct/specific style. This implies that restricting the dataset so that art styles used are more uniform would improve the model’s performance. So, we decided to train the GAN on groups of images from within the same movie genre. 
    
In building a GAN to generate movie posters, our first technical challenge was finding an adequate data set. Datasets could be found with a wide variety of highly diverse images but were not organized by genre and could potentially confuse the networks by offering too many images belonging to wildly different styles. On the other hand it was possible to find datasets of purely a specific genre or time period, but then lacked raw quantity. Moving forward, we expect to encounter potential issues around tuning the learning rate of the NNs relative to one another, as it is possible for the discriminator to train too quickly and fail to provide useful information to the generator. Similarly, it is also entirely possible, as mentioned above, to encounter mode collapse - as a result, we expect to work through these difficulties by adjusting our loss functions and tuning discriminator weights. The validation techniques that we expect to use are quantitative evaluations such as inception score (IS), modified inception score (m-IS), and log likelihood; however, since there is no consensus on an objective evaluation function for GANs, manual evaluation will also be used. 

The final posters generated, we hypothesize, will result in a few distinct, recognizable elements, but generally will struggle with visual specificity. This will be especially true with people and text, elements that are often key to existing movie posters. Furthermore, our generated posters may end up mixing distinct styles from genres with extremely different styles of art - for example, inadvertently combining realistic movie posters with CGI posters or animated posters could create results that are indistinct and unrecognizable from either. This could also be true across genres - for example, highly stylized horror or comedy  - could also create a similar issue of visual ambiguity. 


### Ethical Implications
Additionally, there are also ethical challenges and considerations which we anticipate while conducting our project. For instance, if we train an AI to generate posters based on existing movie posters, it raises the question of where credit should be given. Is the AI artist responsible for the work? Furthermore, how close the results are to the training sets could result in plagiarism. If this occurs, would the subsequent responsibilities lie on the AI or the creators? Is it unfair to make a model that emulates work that real artists have put creative thought and time into?

## Related Works Draft

1. [CAN: Creative Adversarial Networks, Generating "Art" by Learning About Styles and Deviating from Style Norms](https://arxiv.org/abs/1706.07068)

    In "CAN: Creative Adversarial Networks Generating “Art” by Learning About Styles and Deviating from Style Norms", researchers build upon GANs to create what they call a Creative Adversarial Networks (CANs). This paper points out that the nature of a GAN is to produce art that is indiscriminable from the art the discriminator has access to, and thus it is not creative. In order to make it “creative,” this paper proposes a model in which the discriminator portion of the network gives two types of feedback to the generator: whether or not the discriminator thinks the work is art, and a classification of the genre of art the work belongs to. The goal of the CAN is to generate images that the discriminator thinks is art, but that cannot be easily classified into a specific art genre. This causes the outputs to push the boundaries of genre a little (but not too far), and thus be “creative.”

2. [Generative Adversarial Networks (GANs): Challenges, Solutions, and Further Directions](https://dl.acm.org/doi/10.1145/3446374)

    “Generative Adversarial Networks (GANs): Challenges, Solutions, and Further Directions” provides a broad and systematic overview of GANs and their development over time. It begins with a description of what GANs are; namely, two neural networks (a generator and a discriminator) which “compete” with each other by minimizing their own loss and maximizing the other’s until they reach a Nash equilibrium. In training GANs, three major issues arise: mode collapse, non-convergence, and instability. Mode collapse refers to when a generator only outputs samples with limited diversity, which occurs because the small subset of images consistently tricks the discriminator. Non-convergence occurs when the generator and discriminator oscillate during training, rather than converging at an equilibrium as desired. And finally, instability occurs when one of the neural networks becomes more powerful than the other. The more powerful then fails to provide gradients to the less powerful NN, causing learning to come to a standstill. The article pinpoints and explores solutions to several different causes for these issues such as inappropriate architectures, hyperparameters, loss functions, and optimization algorithms. 

3. [A survey of image synthesis and editing with generative adversarial networks](https://ieeexplore.ieee.org/abstract/document/8195348)

    This paper summarizes the different uses of General Adversarial Networks. GANs consist of a generator network which creates images and a discriminator network that determines if the image can be considered real or fake. They can be used to modify an existing image or create new images from a text input. Although basic GANs can create realistic images it is still difficult to generate high resolution images as well as video and 3D models.

4. [Visual Indeterminacy in GAN Art](https://direct.mit.edu/leon/article/53/4/424/96926/Visual-Indeterminacy-in-GAN-Art)

    The author tackles the topic of visual indeterminacy in GAN art - the quality of images to appear real, but upon looking closer, are actually composed of elements that are unrecognizable. Many artworks generated by GAN models have details that can look real, such as human arms or tall skyscrapers, yet these details are not resolved fully and blend ambiguously into the background or into other objects. The author believes that this indeterminacy stems from imperfect generative models, as GANs at the moment are only capable of combining objects, textures, and backgrounds in an "uncanny valley" manner; models are not yet able to generate new compositions of these elements, and as a result, must rely on training images with similar, existing compositions. Subsequently, the resulting images are almost realistic, but fail to achieve certainty. Moving forward, the paper suggests conducting more research into vision neuroscience models and cortical modeling to better understand how visual indeterminacy can be manipulated in GAN artworks.

5. [Modeling Artistic Workflows for Image Generation and Editing](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630154.pdf)

    “Modeling Artistic Workflows for Image Generation and Editing”: The authors of the paper propose a model that generates images in a given art style and image early in the artist's creation process with the intention of producing a completed image in the correct style and revamping the process of creating art. In order to achieve this the authors design a new optimization process with learning based regularization that prevents the model from overfitting and deconstructing the original art style but allows for the model to create new and unique images. This project was completed for Adobe Inc, so the authors had access to numerous images and image drafts, however they minimized variance by restricting their images in their dataset to chairs, anime drawings, and faces.
    
### Project Update 1

- Name and link to the software you will use (or state that you are writing something from scratch).
  - [PyTorch](https://pytorch.org/) (use to make a GAN)
- Name and link to the dataset that you will be using (or state how you will create your own dataset).
  - [41K movie posters from IMDB](https://www.kaggle.com/dadajonjurakuziev/movieposter)
- Provide a high-level overview of the following:
  1. The type of neural network you will use (e.g., fully connected, convolutional, recurrent, etc.)
     - Deep Convolutional Generative Adversarial Network
  2. The shape and type of your inputs (are they three-channel images? sequences of words? a vector of floating-point values? embeddings? etc.)
     - Discriminator (training) input: three-channel poster images
       - Shape of single input: (3, 386, 256)
         - 3 = number of channels
         - 386 = height in px
         - 256 = width in px   
         - (if there are images in a different resolution, we will rescale them)
     - Generator input: random noise
       - May potentially add an extra variable, i.e. genre, and change our network to a conditional GAN 
  3. The shape and type of your outputs (are you performing classification? regression? segmentation?, etc.)
     - We are outputting three-channel images, shape(3, 386, 256)
    
    


