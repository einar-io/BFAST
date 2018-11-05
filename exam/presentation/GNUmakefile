OUTPUT=presentation.pdf
CFLAGS=--output-directory=build -file-line-error -halt-on-error -interaction=nonstopmode -shell-escape
BFLAGS=--output-directory=build 
CC=lualatex
#CC=pdflatex

# https://en.wikibooks.org/wiki/LaTeX/Bibliography_Management#Why_won't_LaTeX_generate_any_output?
$(OUTPUT): *.tex *.bib GNUmakefile
	$(CC) $(CFLAGS) $(@:.pdf=.tex)
	biber    $(BFLAGS) $(basename $@)
	$(CC) $(CFLAGS) $(@:.pdf=.tex)
	$(CC) $(CFLAGS) $(@:.pdf=.tex)

draft: *.tex
	$(CC) $(CFLAGS) $(OUTPUT:.pdf=.tex)
