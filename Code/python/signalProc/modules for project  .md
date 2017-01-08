# Modules to install
- numpy : for array manipulation
- matplotlib=2.0b4 : for plotting
- scipy : for filters
- sklearn : for classification


# Personal notes:
I worked on some basic preprocessing steps today.
Outliers for channesl are computed over the entire data space; not regarding
different conditions.
Does this make sensee? I think so.
In the preprocessing files they provided they do both a spatial and temporal
filter,  I am not sure why and therefore this is not implemented.
Sometimes the program gives an error that does not make sense, it has something
to do with float division.

# outstanding issues:
i need to clean up the file structures such that it becomes more clear
what i am intending to do.
