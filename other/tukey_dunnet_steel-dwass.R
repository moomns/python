library(multcomp)

setwd(dirpath)
data <- read.csv(filepath)

cols <- c("A", "B", "...")
print("TukeyHSD")

#Tukey-Kramer
for (cnt in 1:length(cols)){
  print(cols[cnt])
  #sinkで標準出力先を変更する
  sink(paste("TukeyHSD-path", cols[cnt], ".txt", sep=""))
  print(TukeyHSD(aov(fomula(paste(paste(cols[cnt]), "~hue")), data=data)))
  #再びsinkをすると、標準出力に戻す
  sink()
}

#Dunnet
for (cnt in 1:length(cols)){
  print(cols[cnt])
  sink(paste("Dunnet-path", cols[cnt], ".txt", sep=""))
  amod <- aov(fomula(paste(paste(cols[cnt]), "~hue")), data=data)
  glht.res <- glht(amod, linfct=mcp(timing="Dunnet"))
  print(summary(glht.res))
  sink()
}

#Steel-Dwass tmp
library(NSM3)
sink(paste("Steel-Dwass-path", cols[cnt], ".txt", sep=""))
pSCDFlig(data$risk_calc, data$group, method="Asymptotic")
sink()