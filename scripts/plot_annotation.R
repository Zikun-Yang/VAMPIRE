library(dplyr)
library(tidyr)
library(ggplot2)

color_list = c("#99CC33", "#389d03", "#1e6203",
                "#7a1a97", "#CC3399", "#dd749a",
                "#9e0704", "#FF9900", "#e9d017",
                "#040ba7", "#3399CC", "#c9c1c0")

# config
height = 1
vertical_spacing = 3

# function to generate plot
generate_plot = function(input_file, output_file) {
# read data
annotation = read.table(input_file, sep = "\t", header = TRUE)
annotation = annotation[, c("seq", "length", "start", "end", "motif", "dir")]

# give y index for each seq
seq_list = unique(annotation$seq)
seq2y = setNames(length(seq_list):1, seq_list)

# generate a horizonal line for each seq
lines = data.frame(x = numeric(), y = numeric())
for (seq in names(seq2y)) {
  seq_length = annotation[annotation$seq == seq, "length"][1]
  y = seq2y[seq]
  lines = rbind(lines, data.frame(x = c(0, seq_length - 1), y = c(y, y)))
}

# give color
# rule 1: from most frequent to less
print("give color...")
colormap = list()
motif2index = c()
cur = 1 # max 12

# count motif and sort
motif_count = annotation %>%
                group_by(motif) %>%
                summarise(count = n(), .groups = "drop")
motif_count = motif_count[order(-motif_count$count),]

# assign color
for (idx in 1:nrow(motif_count)){
  motif = as.character(motif_count[idx, "motif"])
  colormap[[motif]] = color_list[[cur]]
  if (cur < 12){ cur = cur + 1}
}


# convert data form
rectangles = annotation %>%
  rowwise() %>%
  summarise(
    seq = seq,
    start = start,
    end = end,
    motif = motif,
    dir = dir,
    x = list(c(start, end, end, start)),
    y = list(
      if (dir == "+") {
        c(seq2y[seq], seq2y[seq], seq2y[seq] + height, seq2y[seq] + height)
      } else {
        c(seq2y[seq], seq2y[seq], seq2y[seq] - height, seq2y[seq] - height)
      }
    )
  ) %>%
  unnest(cols = c(x, y))

# plot
fig = ggplot() +
        geom_line(data = lines, aes(x = x, y = y, group = seq)) +
        geom_polygon(data = rectangles,
                   aes(x = x, y = y,
                       group = interaction(seq, start, end),
                       fill = motif)) +
        scale_fill_manual(values = colormap) +
        labs(
        title = "Custom Rectangles",
        x = "Position",
        y = "Height",
          fill = "Motif"
        ) +
        theme_minimal()

pdf(output_file, height = 10, width = 20)
print(fig)
dev.off()
}

args <- commandArgs(trailingOnly = TRUE)

# check
if (length(args) != 2) {
  stop("missing parameters", call. = FALSE)
}

input_file = args[1]
output_file = args[2]
generate_plot(input_file, output_file)