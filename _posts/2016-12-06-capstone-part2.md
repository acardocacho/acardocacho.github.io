---
layout: post
title: 2016-12-06-capstone-part2.md
---

# Capstone Project Part2

## Problem Statement, goals and success criteria

I will use multiple machine learning methods and compare how well they perform on a single-label text classification task.

The main goal is to reproduce part of my PhD work using state-of-the-art libraries in Python, and be able to access how this area evolved in the past 10 years.

I consider this work will be successful if I am able to reproduce the initial "related work" from my thesis, which at the time took about one year to complete, for this capstone project.  I expect results to be approximately the same as previously published results, and I will even apply some Machine Learning models that I did not use at the time.

## Outline of proposed methods and models

I will use some of the classification methods available in sklearn for this task, including but not limited to: k-NearestNeighbors, NaiveBayes, and SupportVectorMachines.

I will use accuracy to evaluate how well they perform.

I will use public datasets that are actively used in scientific papers, so that these results are comparable to previously published work.

## Identify risks and assumptions

The datasets are publicly available and have been actively used for research work at least in the past ten years.  They can be uploaded to memory easily because they are not too big.  In fact, some of them exist as part of sklearn.datasets.  It's safe to say that the risks are minimal, and that I know my assumptions to be true.

I will download the datasets from my webpage:

http://ana.cachopo.org/datasets-for-single-label-text-categorization

## Create local PostgreSQL database


I created a local PostgreSQL database and uploaded the data to it as requested, but I will not be using it for my work for two main reasons:

- The datasets are small and fit in memory, so I can use Pandas to query them, because I feel more confortable with it.

- These are datasets containing text, and they can be processed much more efficiently as DataFrames (or np.arrays) than as database tables.

## Query, Sort and Clean Data

The data was already clean and available in text files that could be directly read as csv files.

## Create a Data Dictionary

I will use three datasets commonly used for research in single-label text categorization.

Each dataset was available in a file containing one document per line.

Each document is composed by its class and its terms (or words).

Each document is represented by a "word" representing the document's class, a TAB character and then a sequence of "words" delimited by spaces, representing the terms contained in the document.  This can be read as a csv file where each row is a different document, the first column is the document's class and the other columns are the words in the document.

## Perform and summarize EDA

#### 20 Newsgroups

<p>This dataset is a
collection of approximately 20,000 newsgroup documents, partitioned
(nearly) evenly across 20 different newsgroups.  I used the "bydate" version,
because it already had a standard train/test split.  
</p>
<p>Although already cleaned-up, this dataset still had several
attachments, many PGP keys and some duplicates.
</p>
<p>After removing them and the messages that became empty because of
it, the distribution of train and test messages was the following for
each newsgroup:

<table align="center" border="1">
<tbody>
<tr>
<th colspan="4">20 Newsgroups</th>
</tr>
<tr>
<th>Class</th>
<th># train docs</th>
<th># test docs</th>
<th>Total # docs</th>
</tr>
<tr align="right">
<td>alt.atheism</td>
<td>480</td>
<td>319</td>
<td>799</td>
</tr>
<tr align="right">
<td>comp.graphics</td>
<td>584</td>
<td>389</td>
<td>973</td>
</tr>
<tr align="right">
<td>comp.os.ms-windows.misc</td>
<td>572</td>
<td>394</td>
<td>966</td>
</tr>
<tr align="right">
<td>comp.sys.ibm.pc.hardware</td>
<td>590</td>
<td>392</td>
<td>982</td>
</tr>
<tr align="right">
<td>comp.sys.mac.hardware</td>
<td>578</td>
<td>385</td>
<td>963</td>
</tr>
<tr align="right">
<td>comp.windows.x</td>
<td>593</td>
<td>392</td>
<td>985</td>
</tr>
<tr align="right">
<td>misc.forsale</td>
<td>585</td>
<td>390</td>
<td>975</td>
</tr>
<tr align="right">
<td>rec.autos</td>
<td>594</td>
<td>395</td>
<td>989</td>
</tr>
<tr align="right">
<td>rec.motorcycles</td>
<td>598</td>
<td>398</td>
<td>996</td>
</tr>
<tr align="right">
<td>rec.sport.baseball</td>
<td>597</td>
<td>397</td>
<td>994</td>
</tr>
<tr align="right">
<td>rec.sport.hockey</td>
<td>600</td>
<td>399</td>
<td>999</td>
</tr>
<tr align="right">
<td>sci.crypt</td>
<td>595</td>
<td>396</td>
<td>991</td>
</tr>
<tr align="right">
<td>sci.electronics</td>
<td>591</td>
<td>393</td>
<td>984</td>
</tr>
<tr align="right">
<td>sci.med</td>
<td>594</td>
<td>396</td>
<td>990</td>
</tr>
<tr align="right">
<td>sci.space</td>
<td>593</td>
<td>394</td>
<td>987</td>
</tr>
<tr align="right">
<td>soc.religion.christian</td>
<td>598</td>
<td>398</td>
<td>996</td>
</tr>
<tr align="right">
<td>talk.politics.guns</td>
<td>545</td>
<td>364</td>
<td>909</td>
</tr>
<tr align="right">
<td>talk.politics.mideast</td>
<td>564</td>
<td>376</td>
<td>940</td>
</tr>
<tr align="right">
<td>talk.politics.misc</td>
<td>465</td>
<td>310</td>
<td>775</td>
</tr>
<tr align="right">
<td>talk.religion.misc</td>
<td>377</td>
<td>251</td>
<td>628</td>
</tr>
<tr align="right">
<th>Total</th>
<th>11293</th>
<th>7528</th>
<th>18821</th>
</tr>
</tbody>
</table>

#### Reuters 21578

<p>I downloaded the Reuters-21578 dataset from <a href="http://www.daviddlewis.com/resources/testcollections/reuters21578/">David
Lewis' page</a> and used the standard "modApte" train/test split.  These documents
appeared on the Reuters newswire in 1987 and were manually classified
by personnel from Reuters Ltd.  

</p>
<p>Due to the fact that the class distribution for these documents is
very skewed, two sub-collections are usually considered for text
categorization tasks:

</p>
<ul>
<li><strong>R10</strong> The set of the 10 classes with the highest number of 
positive training examples.
</li>
<li><strong>R90</strong> The set of the 90 classes with at least one positive 
training and testing example.
</li></ul>
<p>Moreover, many of these documents are classified as having no topic
at all or with more than one topic.  In fact, you can see the
distribution of the documents per number of topics in the following
table, where <i># train docs</i> and <i># test docs</i> refer to
the <i>Mod Apte</i> split and <i># other</i> refers to documents
that were not considered in this split:

</p>
<table align="center" border="1">
<tbody>
<tr>
<th colspan="5">Reuters 21578</th>
</tr>
<tr>
<th># Topics</th>
<th># train docs</th>
<th># test docs</th>
<th># other</th>
<th>Total # docs</th>
</tr>
<tr align="right">
<td>0</td>
<td>1828</td>
<td>280</td>
<td>8103</td>
<td>10211</td>
</tr>
<tr align="right">
<td>1</td>
<td>6552</td>
<td>2581</td>
<td>361</td>
<td>9494</td>
</tr>
<tr align="right">
<td>2</td>
<td>890</td>
<td>309</td>
<td>135</td>
<td>1334</td>
</tr>
<tr align="right">
<td>3</td>
<td>191</td>
<td>64</td>
<td>55</td>
<td>310</td>
</tr>
<tr align="right">
<td>4</td>
<td>62</td>
<td>32</td>
<td>10</td>
<td>104</td>
</tr>
<tr align="right">
<td>5</td>
<td>39</td>
<td>14</td>
<td>8</td>
<td>61</td>
</tr>
<tr align="right">
<td>6</td>
<td>21</td>
<td>6</td>
<td>3</td>
<td>30</td>
</tr>
<tr align="right">
<td>7</td>
<td>7</td>
<td>4</td>
<td>0</td>
<td>11</td>
</tr>
<tr align="right">
<td>8</td>
<td>4</td>
<td>2</td>
<td>0</td>
<td>6</td>
</tr>
<tr align="right">
<td>9</td>
<td>4</td>
<td>2</td>
<td>0</td>
<td>6</td>
</tr>
<tr align="right">
<td>10</td>
<td>3</td>
<td>1</td>
<td>0</td>
<td>4</td>
</tr>
<tr align="right">
<td>11</td>
<td>0</td>
<td>1</td>
<td>1</td>
<td>2</td>
</tr>
<tr align="right">
<td>12</td>
<td>1</td>
<td>1</td>
<td>0</td>
<td>2</td>
</tr>
<tr align="right">
<td>13</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0</td>
</tr>
<tr align="right">
<td>14</td>
<td>0</td>
<td>2</td>
<td>0</td>
<td>2</td>
</tr>
<tr align="right">
<td>15</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0</td>
</tr>
<tr align="right">
<td>16</td>
<td>1</td>
<td>0</td>
<td>0</td>
<td>1</td>
</tr>
</tbody>
</table>
<p>As the goal in this project is to consider
<strong>single-labeled</strong> datasets, all the documents with less
than or with more than one topic were eliminated.  With this some of
the classes in R10 and R90 were left with no train or test documents.

</p>
<p>Considering only the documents with a single topic and the classes
which still have at least one train and one test example, we have 8 of the
10 most frequent classes and 52 of the original 90.  

</p>
<p>Following Sebastiani's convention, we will call these sets
<strong>R8</strong> and <strong>R52</strong>.  Note that from R10 to
R8 the classes <i>corn</i> and <i>wheat</i>, which are intimately
related to the class <i>grain</i> disapeared and this last class lost
many of its documents.

</p>
<p>The distribution of documents per class is the following for
<strong>R8</strong> and <strong>R52</strong>:

</p>
<table align="center" border="1">
<tbody>
<tr>
<th colspan="4">R8</th>
</tr>
<tr>
<th>Class</th>
<th># train docs</th>
<th># test docs</th>
<th>Total # docs</th>
</tr>
<tr align="right">
<td>acq</td>
<td>1596</td>
<td>696</td>
<td>2292</td>
</tr>
<tr align="right">
<td>crude</td>
<td>253</td>
<td>121</td>
<td>374</td>
</tr>
<tr align="right">
<td>earn</td>
<td>2840</td>
<td>1083</td>
<td>3923</td>
</tr>
<tr align="right">
<td>grain</td>
<td>41</td>
<td>10</td>
<td>51</td>
</tr>
<tr align="right">
<td>interest</td>
<td>190</td>
<td>81</td>
<td>271</td>
</tr>
<tr align="right">
<td>money-fx</td>
<td>206</td>
<td>87</td>
<td>293</td>
</tr>
<tr align="right">
<td>ship</td>
<td>108</td>
<td>36</td>
<td>144</td>
</tr>
<tr align="right">
<td>trade</td>
<td>251</td>
<td>75</td>
<td>326</td>
</tr>
<tr align="right">
<th>Total</th>
<th>5485</th>
<th>2189</th>
<th>7674</th>
</tr>
</tbody>
</table>
<table align="center" border="1">
<tbody>
<tr>
<th colspan="4">R52</th>
</tr>
<tr>
<th>Class</th>
<th># train docs</th>
<th># test docs</th>
<th>Total # docs</th>
</tr>
<tr align="right">
<td>acq</td>
<td>1596</td>
<td>696</td>
<td>2292</td>
</tr>
<tr align="right">
<td>alum</td>
<td>31</td>
<td>19</td>
<td>50</td>
</tr>
<tr align="right">
<td>bop</td>
<td>22</td>
<td>9</td>
<td>31</td>
</tr>
<tr align="right">
<td>carcass</td>
<td>6</td>
<td>5</td>
<td>11</td>
</tr>
<tr align="right">
<td>cocoa</td>
<td>46</td>
<td>15</td>
<td>61</td>
</tr>
<tr align="right">
<td>coffee</td>
<td>90</td>
<td>22</td>
<td>112</td>
</tr>
<tr align="right">
<td>copper</td>
<td>31</td>
<td>13</td>
<td>44</td>
</tr>
<tr align="right">
<td>cotton</td>
<td>15</td>
<td>9</td>
<td>24</td>
</tr>
<tr align="right">
<td>cpi</td>
<td>54</td>
<td>17</td>
<td>71</td>
</tr>
<tr align="right">
<td>cpu</td>
<td>3</td>
<td>1</td>
<td>4</td>
</tr>
<tr align="right">
<td>crude</td>
<td>253</td>
<td>121</td>
<td>374</td>
</tr>
<tr align="right">
<td>dlr</td>
<td>3</td>
<td>3</td>
<td>6</td>
</tr>
<tr align="right">
<td>earn</td>
<td>2840</td>
<td>1083</td>
<td>3923</td>
</tr>
<tr align="right">
<td>fuel</td>
<td>4</td>
<td>7</td>
<td>11</td>
</tr>
<tr align="right">
<td>gas</td>
<td>10</td>
<td>8</td>
<td>18</td>
</tr>
<tr align="right">
<td>gnp</td>
<td>58</td>
<td>15</td>
<td>73</td>
</tr>
<tr align="right">
<td>gold</td>
<td>70</td>
<td>20</td>
<td>90</td>
</tr>
<tr align="right">
<td>grain</td>
<td>41</td>
<td>10</td>
<td>51</td>
</tr>
<tr align="right">
<td>heat</td>
<td>6</td>
<td>4</td>
<td>10</td>
</tr>
<tr align="right">
<td>housing</td>
<td>15</td>
<td>2</td>
<td>17</td>
</tr>
<tr align="right">
<td>income</td>
<td>7</td>
<td>4</td>
<td>11</td>
</tr>
<tr align="right">
<td>instal-debt</td>
<td>5</td>
<td>1</td>
<td>6</td>
</tr>
<tr align="right">
<td>interest</td>
<td>190</td>
<td>81</td>
<td>271</td>
</tr>
<tr align="right">
<td>ipi</td>
<td>33</td>
<td>11</td>
<td>44</td>
</tr>
<tr align="right">
<td>iron-steel</td>
<td>26</td>
<td>12</td>
<td>38</td>
</tr>
<tr align="right">
<td>jet</td>
<td>2</td>
<td>1</td>
<td>3</td>
</tr>
<tr align="right">
<td>jobs</td>
<td>37</td>
<td>12</td>
<td>49</td>
</tr>
<tr align="right">
<td>lead</td>
<td>4</td>
<td>4</td>
<td>8</td>
</tr>
<tr align="right">
<td>lei</td>
<td>11</td>
<td>3</td>
<td>14</td>
</tr>
<tr align="right">
<td>livestock</td>
<td>13</td>
<td>5</td>
<td>18</td>
</tr>
<tr align="right">
<td>lumber</td>
<td>7</td>
<td>4</td>
<td>11</td>
</tr>
<tr align="right">
<td>meal-feed</td>
<td>6</td>
<td>1</td>
<td>7</td>
</tr>
<tr align="right">
<td>money-fx</td>
<td>206</td>
<td>87</td>
<td>293</td>
</tr>
<tr align="right">
<td>money-supply</td>
<td>123</td>
<td>28</td>
<td>151</td>
</tr>
<tr align="right">
<td>nat-gas</td>
<td>24</td>
<td>12</td>
<td>36</td>
</tr>
<tr align="right">
<td>nickel</td>
<td>3</td>
<td>1</td>
<td>4</td>
</tr>
<tr align="right">
<td>orange</td>
<td>13</td>
<td>9</td>
<td>22</td>
</tr>
<tr align="right">
<td>pet-chem</td>
<td>13</td>
<td>6</td>
<td>19</td>
</tr>
<tr align="right">
<td>platinum</td>
<td>1</td>
<td>2</td>
<td>3</td>
</tr>
<tr align="right">
<td>potato</td>
<td>2</td>
<td>3</td>
<td>5</td>
</tr>
<tr align="right">
<td>reserves</td>
<td>37</td>
<td>12</td>
<td>49</td>
</tr>
<tr align="right">
<td>retail</td>
<td>19</td>
<td>1</td>
<td>20</td>
</tr>
<tr align="right">
<td>rubber</td>
<td>31</td>
<td>9</td>
<td>40</td>
</tr>
<tr align="right">
<td>ship</td>
<td>108</td>
<td>36</td>
<td>144</td>
</tr>
<tr align="right">
<td>strategic-metal</td>
<td>9</td>
<td>6</td>
<td>15</td>
</tr>
<tr align="right">
<td>sugar</td>
<td>97</td>
<td>25</td>
<td>122</td>
</tr>
<tr align="right">
<td>tea</td>
<td>2</td>
<td>3</td>
<td>5</td>
</tr>
<tr align="right">
<td>tin</td>
<td>17</td>
<td>10</td>
<td>27</td>
</tr>
<tr align="right">
<td>trade</td>
<td>251</td>
<td>75</td>
<td>326</td>
</tr>
<tr align="right">
<td>veg-oil</td>
<td>19</td>
<td>11</td>
<td>30</td>
</tr>
<tr align="right">
<td>wpi</td>
<td>14</td>
<td>9</td>
<td>23</td>
</tr>
<tr align="right">
<td>zinc</td>
<td>8</td>
<td>5</td>
<td>13</td>
</tr>
<tr align="right">
<th>Total</th>
<th>6532</th>
<th>2568</th>
<th>9100</th>
</tr>
</tbody>
</table>

  

## How to tune and evaluate results

I will evaluate my results using accuracy, the standard evaluation measure for single-label text categorisation tasks.  I am already going to use three versions of two different datasets commonly used for research, and I can compare my results with others that have been previously published.

If I find other public datasets that are commonly used in research papers, I will probably use them as well.


Full details in this notebook in my Github repo:

[Repo](https://github.com/acardocacho/DSI_LDN_1_HOMEWORK/blob/cap-p2/ana/capstone/Ana-Capstone-Part2.ipynb)




