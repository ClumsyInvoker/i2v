
dim = 2
dim_mults = (1, 2, 4, 8)
input_channels = 3

dims = [input_channels, *map(lambda m: dim * m, dim_mults)]

print(dims)