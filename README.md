**About**

A model to predict the overall rating/opinion of a given review text (Part1) and Topic Modelling on commonly mentioned features (for example, important features of a phone could be the battery life, quality of display, touch etc.) using unsupervised techniques (Part2).
The granularity (1-5 or Positive, Neutral, Negative)

Dataset size: **194,439 words**

Number of Unique Products in the Cell Phones and Accesories Category = **10429**

**PART 1**

Sentiment Analysis using Logistic Regression

Number of features: **18994**

Mean cross-validation accuracy: **0.860**

Accuracy:   **0.865**

Training set score: **0.881**

Test set score: **0.865**

Confusion matrix:

[[ 6835  4539]

[ 2025 35211]]

---------------

[['TN' 'FP']

['FN' 'TP']]

Logistic Reg - **F1 score: 0.915**

**PART2**

Topic Modelling using NMF & LDA

Number of total features: **86576**

***Using NMF - TF IDF***

Topic #0:

case protection iphone nice like cases hard fits looks fit otterbox back color buttons plastic

Topic #1:

headset sound ear bluetooth use quality headphones music speaker volume device easy hear also button

Topic #2:

screen protector protectors bubbles easy apply install clear dust put glass get glare scratches touch

Topic #3:

great works price product recommend well fits looks buy would easy perfectly item highly quality

Topic #4:

charger charge usb cable charging car devices power port ipad device plug iphone charges works

Topic #5:

phone cover well fits protects protect dropped fit use put like cell easy back perfectly

Topic #6:

love color cute colors absolutely perfect pink compliments fits really easy awesome recommend super friends

Topic #7:

battery charge batteries life day hours extended original charged last oem pack long extra stock

Topic #8:

good price quality product looks nice really buy pretty protection recommend well cheap fits fit

Topic #9:

one bought would like got get buy another work time really ordered product cheap first

***Using LDA - TF IDF***






