# oscarobber

<img width=200 src="https://polakowo.io/oscarobber/images/stanlee.jpg"/>

#### Graphs are everywhere, especially in an increasingly interconnected world

The idea is to use graph mining to analyze collaboration of actors in science fiction movies, where two actors are nodes connected by an edge if they both appeared in the same movie. For this, we applied a set of social network analysis techniques on a network of 200k actors and 20k sci-fi movies with Python, sklearn and NetworkX. 

Here we assessed the structure and behavior of the network using measures from complex network theory, such as clustering, degree distribution, centrality, assortativity, modularity, growth and preferential attachment. For example: *"What are the most influential actors/movies in the network? How do actors choose in which movies to play? Do some attributes such as genre build good communities? Who is crucial in connecting Bollywood and Hollywood?"* 

We also applied NLP methods to analyze plot summaries, user reviews and the sentiment they express. To make things more acessible, the results of analysis are published on [an interactive website](https://polakowo.io/oscarobber/).
