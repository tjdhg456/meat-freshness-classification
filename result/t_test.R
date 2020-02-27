library(psych)

df = read.csv('result_boxplot.csv', header=TRUE)

# Two dataset
non = df[(df['fusion'] == 'none') & (df['loss'] == 'focal'), 'f1']
fuse = df[(df['fusion'] != 'none') & (df['loss'] == 'focal'), 'f1']

# F-test
var.test(non, fuse) # p > 0.5 --> not significantly different

# t-test (unpaired)
t.test(non, fuse, alternative=c('two.sided'), paired=FALSE, var.equal=TRUE)


