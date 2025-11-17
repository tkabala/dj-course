import { addMatrices, multiplyMatrices, transpose, assertMatricesDimensionMatch, assertMatricesCompatible } from "./matrix-operations";
import { fromJSONFile, jsonFilePath, randomizeMatrix, randomizeVector } from "./utils";
import { vectorSum, dotProduct } from "./vector-operations";
import { Matrix, Vector } from "./types";
import { displayVector, displayMatrix } from "./display";

// HINT: (w zaleności od wybranego kierunku implementacji) może być mnożenie macierzy przez wektory - tę operację będzie trzeba zaimplementować 😉 
// ale nie jest to konieczne 😎

// HINT: w mnożeniu macierzy kolejność ma znaczenie - bo w zależności od kolejności albo wymiary obydwu składników pasują do siebie albo nie.

// HINT: wstań od komputera i przemyśl problem. Serio. Zastanów się, ile linijek wystarczy aby podać rozwiązanie :)
// (traktując "linijkę" jako pojedynczą operację na tensorach) 😎

// PROŚBA: jeśli znasz rozwiązanie, to nie spamuj discorda - a przynajmniej nie od razu. Pozwól innym pomóżdżyć 😎

// przypomnienie zadania: naley policzyć "attention matrix S"

for (let caseNum = 1; caseNum <= 4; caseNum++) {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`CASE ${caseNum}`);
    console.log('='.repeat(60));

    const { WK_Matrix, WQ_Matrix, X_Input_Matrix } = fromJSONFile(jsonFilePath(`case-${caseNum}.json`));

    var Q_Matrix = multiplyMatrices(X_Input_Matrix, WQ_Matrix)
    var K_Matrix = multiplyMatrices(X_Input_Matrix, WK_Matrix)
    var S_Matrix = multiplyMatrices(Q_Matrix, transpose(K_Matrix))
    console.log('S_Matrix');
    console.log(displayMatrix(S_Matrix, -1));
}