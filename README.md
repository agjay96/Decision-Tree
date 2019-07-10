# Decision-Tree

For Decision Tree, we will implement ID3 algorithm. It's guaranteed that all features are discrete.

In ID3 algorithm, we use Entropy to measure the uncertainty in the data set. We use Information Gain to measure the quality of a split.
Entropy: H(S)= ∑x∈X−p(x)log2p(x)∑x∈X−p(x)log2p(x) 
Information_Gain: IG(S,A) = H(S)- ∑t∈Tp(t)H(T)∑t∈Tp(t)H(T)  = H(S) - H(S|A)

<img src="F1Score.png">
