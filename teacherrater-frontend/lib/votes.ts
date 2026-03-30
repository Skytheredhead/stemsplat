export type VoteChoice = "upvote" | "downvote";

export interface VoteState {
  upvotes: number;
  downvotes: number;
  userVote: VoteChoice | null;
}

interface VoteRecord {
  upvotes: number;
  downvotes: number;
  votesByUser: Record<string, VoteChoice>;
}

const VOTES_KEY = "ids-teachers-votes";

function loadAll(): Record<string, VoteRecord> {
  if (typeof window === "undefined") {
    return {};
  }
  const raw = window.localStorage.getItem(VOTES_KEY);
  if (!raw) {
    return {};
  }
  try {
    return JSON.parse(raw) as Record<string, VoteRecord>;
  } catch {
    return {};
  }
}

function saveAll(data: Record<string, VoteRecord>): void {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.setItem(VOTES_KEY, JSON.stringify(data));
}

export function loadVoteState(reviewId: number, userId: number): VoteState {
  const data = loadAll();
  const record = data[String(reviewId)] ?? { upvotes: 0, downvotes: 0, votesByUser: {} };
  return {
    upvotes: record.upvotes,
    downvotes: record.downvotes,
    userVote: record.votesByUser[String(userId)] ?? null,
  };
}

export function applyVote(
  reviewId: number,
  userId: number,
  nextVote: VoteChoice
): VoteState {
  const data = loadAll();
  const key = String(reviewId);
  const record = data[key] ?? { upvotes: 0, downvotes: 0, votesByUser: {} };
  const userKey = String(userId);
  const previous = record.votesByUser[userKey];

  if (previous === nextVote) {
    return {
      upvotes: record.upvotes,
      downvotes: record.downvotes,
      userVote: previous,
    };
  }

  if (previous === "upvote") {
    record.upvotes = Math.max(0, record.upvotes - 1);
  }
  if (previous === "downvote") {
    record.downvotes = Math.max(0, record.downvotes - 1);
  }

  if (nextVote === "upvote") {
    record.upvotes += 1;
  } else {
    record.downvotes += 1;
  }

  record.votesByUser[userKey] = nextVote;
  data[key] = record;
  saveAll(data);

  return {
    upvotes: record.upvotes,
    downvotes: record.downvotes,
    userVote: nextVote,
  };
}
