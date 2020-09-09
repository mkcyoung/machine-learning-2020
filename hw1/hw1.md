# CS 6350: Machine Learning Fall 2020

## Homework 1

## Michael Young / u0742759

***

## 1. Decision trees

### 1. 	Boolean functions -> decision trees		![Screen Shot 2020-09-05 at 4.59.57 PM](/Users/myoung/Desktop/Screen Shot 2020-09-05 at 4.59.57 PM.png)

d continued) The efficiency of using a decision tree to represent m-of-n functions depends on the relationship between m and n. If m is either very low, or very high with respect to n, then there could be some reasonable information gain happening between levels of the tree. But if m is somewhere in the middle, the tree can balloon with several nodes at each level offering not much in the way of information. So I would say that it depends, but in general using decision trees to represent m-of-n functions is not a great idea.

### 2. To wait or not to wait? 

#### a) How many possible functions are there to map these four features to a boolean decision? How many functions are consistent with the given training dataset?

There are $ 2*2*3*4 = 48 $ possible outputs over these four features, and each output is binary, meaning there are a total of $ 2^{48} = 281,474,976,710,656 $ functions possible to map these four features to a boolean decision. We are given 9 instances with their corresponding labels (meaning we know 9 rows of the 48 rows of the truth table). Given that we know these 9 rows, that means that there are $ 2^{48-9} = 2^{39} = 549,755,813,888 $ functions that are still consistent with the given training dataset. 

#### b) What is the entropy of the labels in this data? When calculating entropy, the base of the logarithm should be base 2.

There are 5 Yes labels in this dataset, and 4 No labels. Therefore, 
$$
Entropy(S) = H(S) = -p_{+}log_{2}(p_+) - p_-log_2(p_-) \\
= -\frac{5}{9}log_2(\frac{5}{9}) - \frac{4}{9}log_2(\frac{4}{9}) \\
\approx 0.991
$$

#### c) What is the information gain of each of the features?

Information gain is defined as:
$$
Gain(S,A) = Entropy(S) - \sum_{v\in Values(A)}{\frac{|S_v|}{|S|}}Entropy(S_v) \\
$$
where S is the entire set, and A is the attribute we are splitting on. Starting first with the "Friday" feature, we see that there are two values it could take: Yes or No. $ \frac{6}{9}$ examples belong to"No", and within those 6, they are split between 4 Yes and 2 No. $ \frac{3}{9}$ examples belong to "Yes", and within those 3, 2 are labeled Yes and 1 is labeled No. Thus:
$$
Gain(S,Friday) = Entropy(S) - \frac{3}{9}Entropy(S_{Friday->Yes}) + \frac{6}{9}Entropy(S_{Friday->No})\\
 = Entropy(S) - \frac{3}{9}(-\frac{1}{3}log_2(\frac{1}{3}) - \frac{2}{3}log_2(\frac{2}{3})) + \frac{6}{9}(-\frac{4}{6}log_2(\frac{4}{6}) - \frac{2}{6}log_2(\frac{2}{6})) \\
 \approx 0.0728
$$
Proceeding in a similar fashion for all features, we see:
$$
Gain(S,Hungry) \approx 0.2294 \\
Gain(S,Patrons) \approx 0.6305 \\
Gain(S,Type) \approx 0.1567
$$

#### d) Which attribute will you use to construct the root of the tree using the ID3 algorithm?

The ID3 algortithm will select the root attribute based on which one reduces the label entropy the most, i.e. the one that has the highest information gain. Based on the calculations made in the previous part, this attribute is *Patrons*.

#### e) Using the root that you selected in the previous question, construct a decision tree that represents the data. You do not have to use the ID3 algorithm here, you can show any tree with the chosen root.

![Screen Shot 2020-09-05 at 7.56.00 PM](/Users/myoung/Desktop/Screen Shot 2020-09-05 at 7.56.00 PM.png)

#### f) Suppose you are given three more examples, listed in table 2. Use your decision tree to predict the label for each example. Also report the accuracy of the classifier that you have learned.

Example 1: Patrons: Full -> Hungry: Yes -> Friday: Yes -> Label: **YES**

Example 2: Patrons: None -> Label: **NO**

Example 3: Patrons: Full -> Hungry: Yes -> Friday: Yes -> Label: **YES**

$ Accuracy = \frac{2}{3}$

### 3. Exploring other impurity measures

#### a) Missclassification

##### i) Write down the definition of the information gain heuristic that uses the misclassification rate as its measure of impurity instead of entropy.

Misclassification is defined as $ Misclassification(S) = 1 - \max_{i}{p_i} $ where $p_i$ is the fraction of examples that have label $i$ and the maximization is over all labels.  The information gain heuristic using misclassification as an impurity measure instead of entropy would be: 
$$
Gain(S,A) = Misclassification(S) - \sum_{v\in Values(A)}{\frac{|S_v|}{|S|}}Misclassification(S_v) \\
$$

##### ii) Use your new heuristic to identify the root attribute for the data in Table 1.

In order to identify the root, we must compute the information gain across each split as we did when entropy was our heuristic. Starting first with the "Friday" feature, we see that there are two values it could take: Yes or No. $ \frac{6}{9}$ examples belong to"No", and within those 6, they are split between 4 Yes and 2 No. $ \frac{3}{9}$ examples belong to "Yes", and within those 3, 2 are labeled Yes and 1 is labeled No. Thus:
$$
Gain(S,Friday) = Misclassification(S) - \frac{3}{9}Misclassification(S_{Friday->Yes}) + \frac{6}{9}Misclassification(S_{Friday->No})\\
 = 1 - \max({\frac{5}{9},\frac{4}{9}}) - \frac{3}{9}(1 - \max({\frac{1}{3},\frac{2}{3}})) + \frac{6}{9}(1 - \max({\frac{4}{6},\frac{2}{6}})) \\
 = \frac{1}{9}
$$
Likewise for splitting on other features we get:
$$
Gain(S,Hungry) \approx 0.222 \\
Gain(S,Patrons) \approx 0.333 \\
Gain(S,Type) \approx 0.111
$$
Based on this heuristic, we would select "patrons" as our root attribute, just as in the case when entropy was our heuristic. 

#### b) Another heuristic that is used to define impurity is the Gini coefficient, which is defined as $ Gini(S) = \sum_{i}{p_i(1 - p_i)} $. Use Gini coefficient to identify the root attribute for the training data in Table 1.

Similarly for misclassification and entropy, using Gini gives us the gain formula:
$$
Gain(S,A) = Gini(S) - \sum_{v\in Values(A)}{\frac{|S_v|}{|S|}}Gini(S_v) \\
$$
For splitting on Friday, this looks like:
$$
= \frac{5}{9}(1-\frac{5}{9})+\frac{4}{9}(1-\frac{4}{9}) - (\frac{3}{9}(\frac{1}{3}(1-\frac{1}{3})+\frac{2}{3}(1-\frac{2}{3})) + \frac{6}{9}(\frac{2}{6}(1-\frac{2}{6})+\frac{4}{6}(1-\frac{4}{6}))) \\
= \frac{4}{81} \\
\approx 0.0494
$$
Proceeding this way for the rest we get:
$$
Gain(S,Hungry) \approx  0.149 \\
Gain(S,Patrons) \approx 0.327 \\
Gain(S,Type) \approx 0.086
$$
Once again, according to this metric, we would use "Patrons" as our root note because it gives us the highest information gain. 

