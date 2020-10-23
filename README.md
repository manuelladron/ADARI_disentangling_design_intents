# Multimodal Machine Learning: Cross-modal retrieval on ADARI dataset

Repo for the course 11-777 Multimodal Machine Learning, at Carnegie Mellon University. 

## Introduction
Design language is ambiguous; many terms and expressionsconvey the same idea, while many vastly different objectsare created with the same design intents. Despite advancesin multimodal research on captioning and retrieval tasks [1-7], subjective reasoning and description remain unresolved.Multimodal machine learning can describe an image of “agirl wearing a red shirt” but is unable to describe or generatea “minimal, elegant and sleek looking chair.” For compu-tational linguistics, resolving this recognized problem canchallenge the status of theoretical understanding, problem-solving methods and evaluation techniques [8].  For com-puter vision, this presents a complex challenge of disentan-gling qualitative attributes—sleek, elegant, minimal—fromimages. Beyond its relevance in pushing machine learningresearch boundaries, this would significantly impact creativepractice —designers, architects and engineers.  Real-timedesign intents understanding could open new design scenar-ios (e.g. voice-assisted natural language input), that reduceprocedures based on intent reinterpretation as imperativecommands —move, circle, radius, extrude, vertical— re-quired by digital design engines.

Our project gravitates around the Ambiguos Descriptionsand Art Images—ADARI—dataset. This dataset is a novelcollection of images and descriptions with special emphasison qualitative attributes present in design intents. The nov-elty of this project comes, therefore, from the study of suchsubjective qualitative attributes, and currently, we are testingwhether state-of-the-art multimodal approaches on retrieval,captioning or generation work on ADARI. We prove thatsuch current models, objective functions and evaluation met-rics are not efficient when dealing with intents, and that weare initiating an exploratory process to determine which taskwe will focus on for the remainder of the project. 

### ADARI Dataset 
To address the issue of disentangling design intents in thecontext of creative practice,  we use the Ambiguous De-scriptions and Art Images—ADARI—dataset.   The self-annotated ADARI dataset contains a total of 33,230 samplesof contemporary creative works represented by 264,028 raw sentences provided by the original creators and byart curators that describe 241,982 images.  This datasetwas assembled by collecting articles that include editorialdescriptions along with associated images of the creativevisual work.  ADARI is an organized dataset divided intoseven categories: architecture, art, design, fashion, furniture,product design and technology. ADARI is the first datasetof pairs of images and descriptions that, besides containingobjective information of the elements in the images such as wooden chair or black table, focuses on descriptions that correspond to design intents, such as minimal, elegant and sleek looking chair. Table 1 shows the raw counts of samples, images per domain, and sentences and adjectives fromcreatorsandcuratorsper domain. Table 2 shows that givena visual workpiece, original creators use a more ambiguouslanguage than curators in their descriptions, that it, normalized results of a human classification of ADARI labels intoqualitative or quantitative attributes.  Over 500 adjectivesper source (cretaor, curators and both) and per domain (furniture, fashion, wearable technology) have been annotated.Unsurprisingly, creators express their design intents whilecurators tend to describe the work objectively. Both sourcesof language are used as annotations for the labels of thedataset.

<div  align="center">   
  <img width="70%"   src="./media/adari.png">
  <p style="font-size:10px"> Figure 1.ADARI dataset. Rows: architecture (top), furniture (mid-dle), fashion (bottom) sub-domains in ADARI. Columns, from left to right: images with label <em>interesting, simple, organic, different,iconic, minimalist.</em> </p>
</div>

To simplify the complexity of the ADARI datset, and unlessotherwise stated, we use the ADARI Furniture domain inall  experiments.   This  contains  over  17,000  images  andapproximately a total of 60,000 adjectives.

<div  align="center">   
  <img width="50%"   src="./media/ADARI_raw_statistics.png">
</div>


### Research Challenges
Creative practice is a broad term that encompasses those fields that produce creative work. Specifically, in the contextof this project,  we refer as creative practice to any fieldthat relies on visual mediums and feedback in the creative process.  This includes works by designers, architects or engineers. In these domains, expressive language, used todescribe, review and discuss work, is crucial.  This type of language gravitates around design intents. For example,image a designer says to a colleague that they shoulddesignthe chair with a more dynamic look; the word <em>dynamic</em> conveys the design intent,  but it can be embodied in an intractable number of visual forms and design variations. While  design  intents  also  include  quantitative  attributes, such as colors, materials or dimensions, a great part of thevocabulary corresponds to qualitative traits.

The main technical challenge of this project is to disentanglesuch attributes from the design objects. For this, tasks suchas classification, retrieval or generation might require different objective functions and evaluation metrics. An important part of this project is to test such different techniques andcurrent models on ADARI, and gauge their effectiveness in disentangling intents.


<div  align="center">   
  <img width="50%"   src="./media/ADARI_ambiguity_and_relevance.png">
</div>



## Prior Work
This section first reviews general previous work that provide some context and frame this project.  We then explain in detail some prior work that we have implemented and tested on the ADARI dataset. Lastly, we discuss prior work done using the ADARI dataset.

### General Review and Project Context
Powerful representation ability of multimodal approaches with multiple levels of attraction has proven successful in applications on event detection [9, 10], cross-modal retrieval [1, 11–13], cross-modal embeddings [4, 14–16] and image caption [6, 17]. Generative neural models have gained interest and exponential adoption by creative researchers. In addition to offering image generation of unseen but realistic images [18], image painting [19, 20] or image manipulation [21–25]. 

Most recently, generative adversarial learning has extended to multimodal cases, including text-to-image synthesis [26, 27], handwritten text generation [28], and visual captioning [29-31]. This work, directly and indirectly, enable more intelligent and intuitive interfaces for designers: by advancing techniques for object selection, image reconstruction or real time video object removal, used in commercial design software.

While current state-of-the-art image captioning is able to describe an image of <em>“a girl wearing a red shirt”</em>, likewise, current retrieval models are able to retrieve pictures of red shirts. However, neither image captioning systems, retrieval models or generative models are able to describe, retrieve or generate an image of <em>“a sleek, elegant and minimal looking chair”</em>. Our current research project navigates these mediums, aiming to recognize the patterns of human intents that design objects register in the image space. Related to such subjective patterns, Neuberger et al., explore the means to learn subjective fashion traits [32].

### Cross-modal Retrieval Methods
#### Coordinated Representation Baseline: Two Way Nets
ADARI raises a key question about design: How can we unify the intent of the designer with the design itself? Such question immediately leads to coordinated representation learning. This task aims to embed images and the corresponding text in the same subspace. Published in 2017 by Eisenschtat et. al, the Two Way Net architecture uses two tied autoencoders to first encode each view into an embedding, and then transform each view into the opposite one. The authors introduced some novelties, including regularizing the batch normalization layers and tying the masks of dropout layers between paired autoencoders. 

The Two Way Net architecture requires the inputs to be pre-encoded by some unimodal method, ideally one that extracts relevant features for the multimodal case. To evaluate this model, we employed the ResNet embeddings of the images. For the adjectives used to describe each design, we use GloVe embeddings of each word and concatenated them together (as was done in the original Two Way Net paper). We used 3 hidden layers per autoencoder, of sizes 2000, 3000, 16000, with the second layer output used as the hidden embedding (as was done with the Flickr30k experiments). The model was trained from scratch with ADARI, using similar hyperparameter settings to those used for the Flickr30k experiments done by the original authors. The model's loss did converge after 100 epochs, as recommended by the authors of the original paper. As a baseline, we aimed to simply see if Two Way Nets with ResNet and GloVe embeddings were capable of clustering around design intent across modalities. As a simple qualitative test, we used t-SNE to reduce the hidden embeddings of the test set to two dimensions, and plotted the result. We ran t-SNE for all perplexities between 5 and 50, achieving best performance at perplexity 36. 

This clustering suggests the Two Way Nets were able to encode similarities in the unimodal representations, however. A closer inspection of these clusters using simple nearest neighbours shows some relationships like the figures 3 and 4. These chairs share design elements, including both their material and simplicity.

<div  align="center">   
  <img width="15%"   src="./media/chair_a.png">
    <img width="15%"   src="./media/chair_b.png"> 

  <p style="font-size:10px"> Image Nearest Neighbor </p>
</div>

For a quantitative measure of the clustering of our hidden coordinated representations, we performed numerous clustering techniques, including kmeans, affinity propagation, agglomerative clustering, OPTICS, and Birch. However, each method produced only two clusters, each containing one of the modalities. It is thus evident that vanilla Two Way Nets using our configurations are not capable of capturing understanding of design intents. While our analysis is not exhaustive, this result is to be expected given the complexities of design intents. It is highly unlikely out of the box image embedding techniques will not capture features relevant for analysing design intents, as these models are often trained for object detection tasks. This shows the gap between tasks that ADARI presents and classic image captioning/object detection techniques. 


<div  align="center">   
  <img width="28%"   src="./media/tsne_perplexity_36.png">
  <p style="font-size:10px"> Figure 2. Two Way Net-generated embeddings reduced to dimension 2. Green samples are embeddings of images, blue are embeddings of text. </p>
</div>


#### Joint Representation Baselines
We have explored three related papers that provided state-of-the-art results on image and sentence retrieval using joint representations at the time of publishing: <em>Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models</em> by R. Kiros et al., <em>DeViSE: A Deep Visual-Semantic Embedding Model</em> by A. Frome et al., and <em>Multimodal Convolutional Neural Networks for Matching Image and Sentence</em> by L. Ma et al. They all base their learning using different types of contrastive loss. 

Ma et al. propose a model based on three convolutional networks coupled end-to-end, one for learning image representations, called image CNN, another one to produce the joint representation of an encoded image representation and word representations, called matching CNN, and an MLP that takes the joint representation and outputs the matching score between image and words, see Figure 4. The image CNN is a pre-trained CNN network (the authors use VGG with 19 weight layers, and the OverFeat network). The core network is the matching CNN, and the authors design different networks that work at different language-level matching relations, such as word-level, phrase-level or sentence-level. Their loss function is defined as eq. \ref{eq1}, where \(s_{match}\) is the output of a the MLP (using a relu nonlinearity). They test their models on Flickr8K, Flickr30K and Microsoft COCO datasets. For evaluation, they report the median rank, and the R@K (with K=1,5,10), which computes the fraction of times the correct result was found among the top K items. 

$$\loss(image,label) = max [0, margin - s_{match}(x_{n}, y{n}) + s_{match}(x_{n}, y{m})$$  <em>(1)</em>

DeViSE and Unifying visual-semantic embeddings approaches use the same loss function as in eq.2, and instead of using another neural network for calculating the matching score, they use the last layer's weights of a pre-trained CNN, such as VGG and others. DeViSE tests their results on ImageNet (ILSVRC) 2012 1K dataset while that Kiros et al. use Flickr8K and Flickr30K. The main different between the two is that DeViSE use pre-trained Glove vectors directly, and for sentences they average the word emeddings while that Kiros et al. use an LSTM to get the representation of the sentence. 

$$\loss(image,label) = max [0, margin - t_{label}Mv_{image} + t_{contrast}Mv_{image})$$ <em>(2)</em>

We have implemented the three papers, and while we are still running experiments, and therefore, we do not have evaluation metrics yet. We use the same hyperparameters as the original papers, reducing the batch size to 64 due to memory issues. The inputs to the network are variable size list of single words, capping the limit to words max. Different from the original papers, we use a pretrained ResNet-152 as image embeddings, and set the image embedding size matching the word embedding size, that is, 50 dimensions. As loss functions, besides using eq. 1 and 2, we use cosine similarity between the two embeddings. These models converge quick, in less than 10 epochs, and the initial experiments are not being successful. More work is to be done to show the real capacity of these models on ADARI.  

<div  align="center">   
  <img width="40%"   src="./media/matching_cnn.png">
  <p style="font-size:10px">Figure 5. The m-CNN architecture for matching image and sentence. Image representation is generated by the image CNN. Matching CNN composes words to different fragments of the sentence and learns the joint representation of image and sentence fragments. MLP summarizes the joint representation and outputs the matching score. </p>
</div>

#### Prior Work on ADARI
Preliminary explorations that visually disambiguate vague terms in the context of design have been done by [7]. The authors use a multimodal approach that combines a pretrained convolutional neural network, ResNet-152, to get the representation for images with general word indexes into a common joint subspace. A bidirectional Long Short-Term Memory (biLSTM) decoder—which models the labels co-occurrence information—learns semantic relationships between words and images. Early results are positive and encouraging for several reasons: the baseline presented is able to detect design nuances in the images that relate to ambiguous words such as “curvaceous”, “wobbly”, “linear”, or “slouchy”, where none of the corresponding images necessarily had those labels applied in the ground truth (see Figure 6). This is an indication of a potential approach for understanding ambiguous terms through associations of words-images. 

<div  align="center">   
  <img width="70%"   src="./media/predicted_labels_only3.png">
  <p style="font-size:10px">Figure 5. The m-CNN architecture for matching image and sentence. Image representation is generated by the image CNN. Matching CNN composes words to different fragments of the sentence and learns the joint representation of image and sentence fragments. MLP summarizes the joint representation and outputs the matching score. </p>
</div>

## Unimodal Explorations
### Finetune Glove Vectors
To evaluate unimodal representations of in-domain (design) language, we compared three results: training embeddings from scratch, with random initialization of weights, using pre-trained Glove vectors, and fine-tuning Glove vectors following a skipgram model. For that, we initiate the weights of the skipgram model with the glove vectors, and run the model for 20 epochs on ADARI. We found that fine-tuning Glove embeddings yielded better results.  

The vocabulary focuses on 3400 adjectives extracted from both sources, creators and curators in ADARI, capturing both qualitative and quantitative adjectives, plus the existing vocabulary in Glove. If a particular word from ADARI does not exist on the Glove vocabulary, we initiate the weights with a normal distribution with scale of 0.6. We frame the context of the words with a window-size of five on each side of the word, and we tested 50, 100, 200 and 300 dimensional embeddings. The 50 dimensional embeddings yielded better results, evaluated through manual inspection by design experts, see table 1.

<div  align="center">   
  <img width="50%"   src="./media/table2.png">


### DCGAN on ADARI
We ran a DCGAN model on ADARI, using the same parameters as the original paper, to evaluate whether the ADARI dataset provides the means for deep learning models to learn the image distribution. We do not have evaluation metrics yet of this experiment, although the results generated by the network are positive, see Figure \ref{gan}. The fact that ADARI seems consistent across its subdomains, like furniture, is an indicative that we could further use some type of conditional GANs to generate visualizations of design intents. It is important to clarify that we are not looking to generate new design outputs within a subdomain, like a chair, or a table. Rather, we would be interested in creating a model that is able to learn the distributions responsible for shaping design intents, that is, would it be possible to generate images that directly forge the visual attributes corresponding to intents such as dynamic, smooth or minimalist?


<div  align="center">   
  <img width="100%"   src="./media/realvsfake_noline.png">
  <p style="font-size:10px">Figure 7. Results of DCGAN trained on ADARI. The first six columns are real images. </p>
</div>

