#! /bin/bash

# Builds the paper.tex file complete with bibliography.

# Convert all eps files to pdf

outpdf_name=Esterer_Nicholas_WASPAA_2017.pdf
[[ $BUILD_FOR_ARCHIVX ]] && outpdf_name=Esterer_Nicholas_archivx.pdf

# set BUILD_FOR_ARCHIVX=1 for formatting for Archivx
paper_build ()
{
    echo '\def\buildforarchivx{'"$BUILD_FOR_ARCHIVX"'} \input{paper.tex}'
    pdflatex '\def\buildforarchivx{'"$BUILD_FOR_ARCHIVX"'} \input{paper.tex}'
}
paper_build
BIBINPUTS=. bibtex paper.aux
paper_build
paper_build
mv paper.pdf "$outpdf_name"
#dvipdf thesis.dvi
