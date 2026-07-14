export type CorpusDocument = {
  id: string;
  title: string;
  body: string;
};

export type CorpusSearchResult = CorpusDocument & {
  score: number;
};

export function searchCorpus(
  corpus: readonly CorpusDocument[],
  query: string,
): CorpusSearchResult[] {
  const terms = query
    .trim()
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean);

  if (terms.length === 0) {
    return [];
  }

  return corpus
    .map((document) => {
      const haystack = `${document.title}\n${document.body}`.toLowerCase();
      const score = terms.reduce(
        (count, term) => count + (haystack.includes(term) ? 1 : 0),
        0,
      );

      return {
        ...document,
        score,
      };
    })
    .filter((result) => result.score > 0)
    .sort((left, right) => right.score - left.score);
}
