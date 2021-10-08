
# Change the paths to be relative when copying the readme into the docs,
# so the links are correct.
sed  's/resources\/docs\///g' README.md > ./resources/docs/README.md
cp CONTRIBUTING.md ./resources/docs/
cp CODE_OF_CONDUCT.md ./resources/docs/

