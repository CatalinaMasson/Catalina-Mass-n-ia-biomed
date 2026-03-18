import w2v from 'word2vec';
import * as os from 'os';
import * as path from 'path';
import * as fs from 'fs';
const corpusPath = path.resolve(process.cwd(), 'src/ejemplo/corpus.txt');
const modelPath = path.resolve(process.cwd(), 'src/ejemplo/word2vec/vectors.txt');

/**
 * Trains a Word2Vec model from the corpus.txt and saves it to vectors.txt
 */
export async function train() {
    return new Promise<void>((resolve, reject) => {
        w2v.word2vec(corpusPath, modelPath, {
            cbow: 1, // Use Continuous Bag of Words (1 = CBOW, 0 = Skip-Gram)
            size: 5, // Vector dimension size 
            window: 2, // Max skip length between words
            hs: 0, // Use Hierarchical Softmax
            sample: 1e-3, // Threshold for occurrence of words
            threads: 1, // Number of threads
            iter: 50, // Number of training iterations
            minCount: 1, // Minimum word count (important for tiny datasets)
            binary: 0 // Output text vectors instead of binary
        }, (error) => {
            if (error) {
                console.error("Training failed:", error);
                reject(error);
                return;
            }
            console.log(`Model trained and dynamically saved to ${modelPath}\n`);
            resolve();
        });
    });
}

// Softmax Helper Function
export function softmax(scores: number[]): number[] {
    const max = Math.max(...scores); // helps numerical stability
    const exps = scores.map(s => Math.exp(s - max));
    const sumOfExps = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sumOfExps);
}

/**
 * Loads the trained vectors.txt model and evaluates nearest neighbors for "cat" and "dog"
 */
export async function evaluate() {
    return new Promise<void>((resolve, reject) => {
        w2v.loadModel(modelPath, (error, model) => {
            if (error) {
                console.error("Error loading model:", error);
                reject(error);
                return;
            }

            const vocab = model.getVectors().map(v => v.word);
            console.log("Vocabulary:", vocab.join(', '));
            console.log("Vocabulary Size:", vocab.length);

            console.log("\n--- Vector representation of 'cat' ---");
            console.log(model.getVector('cat'));

            console.log("\n--- Nearest Neighbors to 'cat' ---");
            const catVec = model.getVector('cat');
            if (catVec) {
                const nearestCat = model.getNearestWords(catVec, vocab.length) as { word: string, dist: number }[];
                const probabilities = softmax(nearestCat.map(n => n.dist));

                nearestCat.forEach((n, i) => {
                    console.log(`${n.word}: Similarity ${n.dist.toFixed(4)} | Probability ${(probabilities[i] * 100).toFixed(2)}%`);
                });
            }

            console.log("\n--- Nearest Neighbors to 'dog' ---");
            const dogVec = model.getVector('dog');
            if (dogVec) {
                const nearestDog = model.getNearestWords(dogVec, vocab.length) as { word: string, dist: number }[];
                const probabilities = softmax(nearestDog.map(n => n.dist));

                nearestDog.forEach((n, i) => {
                    console.log(`${n.word}: Similarity ${n.dist.toFixed(4)} | Probability ${(probabilities[i] * 100).toFixed(2)}%`);
                });
            }

            resolve();
        });
    });
}

/**
 * Loads the trained vectors.txt model and infers missing outer bounds for CBOW inputs.
 */
export async function evaluateCbow() {
    return new Promise<void>((resolve, reject) => {
        w2v.loadModel(modelPath, (error, model) => {
            if (error) {
                console.error("Error loading model:", error);
                reject(error);
                return;
            }

            const vocab = model.getVectors().map(v => v.word);

            console.log("\n--- Demostrando la pérdida de orden secuencial en CBOW ---");
            console.log("Evaluaremos 'sat on' vs 'on sat'.");

            /*
            En un modelo Continuous Bag-Of-Words (CBOW):
            1. Se separan las palabras: ['sat', 'on'] -> vectores V(sat), V(on)
            2. Se promedian los vectores: (V(sat) + V(on)) / 2 = V(context)
            3. Se buscan las palabras más cercanas a ese vector promedio.

            Como la suma y el promedio son operaciones conmutativas:
            (V(sat) + V(on)) / 2 === (V(on) + V(sat)) / 2

            Esto significa que 'sat on' arroja EXACTAMENTE el mismo resultado que 'on sat'.
            ¡El modelo pierde por completo el contexto del orden de las palabras!
            */

            console.log("\n--- Contexto para 'sat on' ---");
            const satOn = model.mostSimilar('sat on', 3) as { word: string, dist: number }[];
            satOn?.forEach(n => console.log(`${n.word}: ${n.dist.toFixed(4)}`));

            console.log("\n--- Contexto para 'on sat' ---");
            const onSat = model.mostSimilar('on sat', 3) as { word: string, dist: number }[];
            onSat?.forEach(n => console.log(`${n.word}: ${n.dist.toFixed(4)}`));

            resolve();
        });
    });
}



export async function main() {
    try {
        if (os.platform() !== 'win32') {
            try {
                await train();
            } catch (trainError: any) {
                console.warn("Training failed:", trainError?.message || trainError);
            }
        } else {
            console.warn("Skipping training on Windows (requires Make/GCC). Using pre-existing vectors.txt...");
        }
        await evaluate();
        await evaluateCbow();
    } catch (error) {
        console.error("Execution failed:", error);
        process.exit(1);
    }
}

// Automatically run main if this file is executed directly 
if (process.argv[1] && process.argv[1].endsWith('index.ts')) {
    main();
}
