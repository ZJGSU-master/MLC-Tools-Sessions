# MLC-Tools-Sessions
Sample multi-label work sessions using disparate software tools

## How to use these files

Here you can find several source file formats, including:

- `.Rmd`: R Notebook
- `.ipynb`: Jupyter Notebook

These files can be loaded with the proper tool: RStudio, Jupyter, etc., to access the source code and run it locally.

In addition, the output produced by these files is provided in two file formats:

- `.md`: Markdown - You can click on them to see the output directly from GitHub (without images)
- `.html`: HTML - Download the file and open it with your usual browser to see the complete output, including images, without having to run the code

## How to run the Java sessions

The Java work sessions can be reproduced into the Java REPL `jshell`, as long as the MULAN and MEKA `.jar` files are accessible from the class path. In order to run these notebooks using Jupyter a Java kernel has to be installed and configured. The following steps show how to do that:

1. Open a terminal window and verify that a recent Java version is installed in the system: `java -version`. Install the latest JDK if necessary. 

2. Download the latest version of the [IJava kernel](https://github.com/SpencerPark/IJava/releases) and follow the instructions to install it, i.e. by issuing the `python install.py` command in the terminal.

3. Download latest versions of the MULAN and MEKA software. Extract the `.jar` files to a common directory.

4. Configure the `IJAVA_CLASSPATH` environment variable, assigning it the path to the folder holding the previous `.jar` files.

5. Launch `jupyter lab` and select the Java kernel. Then, the notebooks can be loaded and run.

   