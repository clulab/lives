# Visualize missed detection items 

# add required packages
require(ggplot2)
require(tidyr)

# set working directory
setwd("/media/jculnan/backup/From LIVES folder/diarized_files/")

# read in files
data <- read.csv("all_missed.csv")

data <- data[data$length > 0.50,]
data <- data[data$speaker=="coach" | data$speaker == "participant",]

# get total num seconds of missed detection
# 12269.554
# total length 23946
all_missed_secs <- sum(data[,"length"], na.rm=TRUE)

# get total num of seconds from missed detection >= 250 ms
# 11570.795
# total length 16712
twofiftyplus <- data[data$length >= .250,]
missed_250plus_secs <- sum(twofiftyplus[,"length"], na.rm=TRUE)

# get total of seconds from missed detection >= 2 sec
# 1556.000 
# total length 490
twoplus <- data[data$length >= 2.0, ]
missed_2plus_secs <- sum(twoplus[, "length"], na.rm=TRUE)


missed_hist <- ggplot(data=data, aes(x=length)) +
  geom_histogram(breaks=seq(0, 5, .25)) + 
  facet_grid(rows=vars(speaker))

annotated <- read.csv("sampled_missed_items_evaluated.csv")
# this is NOT na, so na.omit doesn't work here
annotated <- annotated[annotated$contains != "",]

# get total number of seconds of all items
# 1610.827  
# total length is 863
annotated_secs <- sum(annotated[, "length"], na.rm=TRUE)
# of these, how many seconds are TRUE speech/laughter
spk <- c("<speech>", "<laughter>")
annotated_speech <- annotated[annotated$contains %in% spk,]
# 305.738
# total length is 163
speech_secs <- sum(annotated_speech[, "length"], na.rm=TRUE)


# get total number of seconds for < 2.00
# 268.187 
# total length is 434
undertwo_ann <- annotated[annotated$length < 2.00,]
undertwo_ann_secs <- sum(undertwo_ann[, "length"], na.rm=TRUE)
# of these, how many seconds are TRUE speech/laughter
# 40.481
# total length is 69
speech_undertwo <- annotated_speech[annotated_speech$length < 2.00,]
speech_undertwo_secs <- sum(speech_undertwo[, "length"], na.rm=TRUE)


# get total number of seconds for >= 2
# 1342.641
# total length is 429
twoplus_ann <- annotated[annotated$length >= 2.00,]
twopus_ann_secs <- sum(twoplus_ann[, "length"], na.rm=TRUE)
# of these, how many seconds are TRUE speech/laughter
# 265.257
# total length is 94
twoplus_speech <- annotated_speech[annotated_speech$length >= 2.00,]
twoplus_speech_secs <- sum(twoplus_speech[, "length"], na.rm=TRUE)


by_type <- ggplot(data=annotated, aes(x=contains, fill=contains)) +
  geom_bar() + 
  facet_grid(rows=vars(language))


by_length <- annotated %>%
  group_by(contains) %>%
  summarise(avg_len=mean(length), stdev_len=sd(length), minimum=min(length),
            first=quantile(length, .25), second=quantile(length, .5), 
            third=quantile(length, .75), maximum=max(length))

by_type <- ggplot(data=by_length, aes(x=contains, fill=contains)) + 
  geom_bar(aes(y=mean))

box_and_whisker <- ggplot(data=annotated, aes(x=contains, y=length, fill=language)) + 
  geom_boxplot()

# get only the data between .25 and 2 seconds 
bw_low <- annotated[annotated$length < 2.00,]

box_whisker_low <- ggplot(data=bw_low, aes(x=contains, y=length)) +
  geom_boxplot()

# get only the data 2+ seconds
bw_high <- annotated[annotated$length >= 2.00,]
  
box_whisker_high <- ggplot(data=bw_high, aes(x=contains, y=length)) +
  geom_boxplot()


###################################################
########### TEST OUT OTHER METRICS ################
###################################################

setwd("/media/jculnan/backup/From LIVES folder/diarization_effort")

sil <- read.csv("all_extracted_silence_info.csv")

tgt <- merge(annotated, sil, on=c("name", "start", "end"), all.x=TRUE, all.y=TRUE)

# check proportion with 0 syll are NOT laughter/speech
laughsp <- c("<laughter>", "<speech>")
tgt$ofinterest <- ifelse(tgt$contains %in% laughsp, 1, 0)
tgt$ofinterest <- as.factor(tgt$ofinterest)

# check length vs of interest
bw_length <- ggplot(data=tgt, aes(x=ofinterest, y=length)) +
  geom_boxplot()

# check number of syllables vs of interest
bw_numsylls <- ggplot(data=tgt, aes(x=ofinterest, y=nsyll)) + 
  geom_boxplot()

# check out num-sylls x length vs of interest
bw_syllslen <- ggplot(data=tgt, aes(x=ofinterest, y=interaction(nsyll, length))) +
  geom_boxplot()
syllslen <- ggplot(data=tgt, aes(x=length, y=nsyll, color=ofinterest)) + 
  geom_point()

syllspauselen <- ggplot(data=tgt, aes(x=length, y=interaction(nsyll, npause), color=ofinterest)) +
  geom_point()

syllspauselenplus <- ggplot(data=tgt, 
                            aes(x=length, 
                                y=interaction(nsyll, npause, phonationtime..s.),
                                color=ofinterest)) +
  geom_point()

# this spilts the data a little bit better than the previous ones
syllspauselenplus <- ggplot(data=tgt, 
                            aes(x=interaction(length, nsyll), 
                                y=interaction(npause, phonationtime..s.),
                                color=ofinterest)) +
  geom_point()

# try doing a PCA
require(stats)
require(ggfortify)

# do pca of four factors
tgt_sm <- tgt[c("length","nsyll","npause", "phonationtime..s.", "ofinterest",
                "pauseprop..pausetime.speechtime.")]
tgt_sm <- tgt[c(4, 11:18, 21)]
tgt_sm <- na.omit(tgt_sm)
pca <- prcomp(tgt_sm[c(8:9)], center=TRUE, scale.=TRUE)
autoplot(pca, data=tgt_sm, colour="ofinterest")


x <- glm(ofinterest ~ articulation.rate..nsyll...phonationtime. +
          speechrate..nsyll.dur., data=tgt_sm,
         family = binomial(link='logit'))

# plot this logistic regression
lrplot <- plot(ggpredict(x, c("articulation.rate..nsyll...phonationtime.", 
                    "speechrate..nsyll.dur.")))

tgt_sm$ofinterest <- as.numeric(tgt_sm$ofinterest)
tgtplot <- ggplot(data=tgt_sm, aes(x=articulation.rate..nsyll...phonationtime.,
                                   y=ofinterest)) +
  geom_point()


tgtplot <- ggplot(data=tgt_sm, aes(x=interaction(speechrate..nsyll.dur., 
                                                 articulation.rate..nsyll...phonationtime.
                                                 ),
                                   y=ofinterest)) +
  geom_point()


# get only the data that did not get values assigned by the silence praat
silent <- tgt[is.na(tgt$npause),]
