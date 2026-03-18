declare module 'word2vec' {
    export interface TrainOptions {
        cbow?: number;
        size?: number;
        window?: number;
        hs?: number;
        sample?: number;
        threads?: number;
        iter?: number;
        minCount?: number;
        binary?: number;
    }

    export interface Vector {
        word: string;
        values: Float32Array;
    }

    export interface NearestWord {
        word: string;
        dist: number;
        values?: Float32Array;
    }

    export interface Model {
        getVectors(): Vector[];
        getVector(word: string): FloatArray | null | undefined;
        getNearestWords(vector: FloatArray, num?: number): NearestWord[];
        mostSimilar(word: string, num?: number): NearestWord[];
    }

    type FloatArray = Float32Array | number[];

    export function word2vec(
        input: string,
        output: string,
        options: TrainOptions,
        callback: (error: Error | null) => void
    ): void;

    export function loadModel(
        modelFile: string,
        callback: (error: Error | null, model: Model) => void
    ): void;
}
