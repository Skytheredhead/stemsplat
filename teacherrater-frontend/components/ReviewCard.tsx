import { useEffect, useState } from "react";
import { ArrowBigDown, ArrowBigUp, Star } from "lucide-react";

import { loadToken, loadUser } from "@/lib/auth";
import { addNotification } from "@/lib/notifications";
import { useSchool } from "@/components/SchoolProvider";
import { Review, UserProfile } from "@/lib/types";
import { applyVote, loadVoteState, VoteChoice, VoteState } from "@/lib/votes";

function Stars({ value }: { value: number }) {
  return (
    <div className="flex items-center gap-1">
      {Array.from({ length: 5 }).map((_, index) => (
        <Star
          key={index}
          className={`h-4 w-4 ${
            index < value ? "fill-theme-primary text-theme-primary" : "text-slate-300"
          }`}
        />
      ))}
    </div>
  );
}

export function ReviewCard({ review }: { review: Review }) {
  const { school } = useSchool();
  const [user, setUser] = useState<UserProfile | null>(null);
  const [voteState, setVoteState] = useState<VoteState>({
    upvotes: 0,
    downvotes: 0,
    userVote: null,
  });

  useEffect(() => {
    const updateUser = () => {
      setUser(loadUser<UserProfile>());
    };
    updateUser();
    window.addEventListener("auth-updated", updateUser);
    return () => window.removeEventListener("auth-updated", updateUser);
  }, []);

  useEffect(() => {
    if (!user) {
      return;
    }
    setVoteState(loadVoteState(review.id, user.id));
  }, [review.id, user]);

  async function handleVote(choice: VoteChoice) {
    if (!user) {
      return;
    }
    const updated = applyVote(review.id, user.id, choice);
    setVoteState(updated);
    if (user.id !== review.user_id) {
      const token = loadToken();
      if (!token) {
        return;
      }
      try {
        await addNotification(school, token, { review_id: review.id, type: choice });
      } catch {
        return;
      }
    }
  }

  const totalVotes = voteState.upvotes + voteState.downvotes;
  const upvotePercentage = totalVotes > 0 ? Math.round((voteState.upvotes / totalVotes) * 100) : 0;

  return (
    <div
      id={`review-${review.id}`}
      className="rounded-2xl p-5 glass-panel"
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-sm text-slate-500">Overall</p>
          <Stars value={review.rating} />
        </div>
        <div className="text-sm text-slate-600">
          <span className="font-semibold">Difficulty:</span> {review.difficulty}
        </div>
        <div className="text-sm text-slate-600">
          <span className="font-semibold">Clarity:</span> {review.clarity}
        </div>
        <p className="text-xs text-slate-400">
          {new Date(review.created_at).toLocaleDateString()}
        </p>
      </div>
      {user && (
        <div className="mt-3 flex items-center gap-3 text-sm text-slate-600">
          <button
            type="button"
            onClick={() => void handleVote("upvote")}
            className={`rounded-full border px-2 py-1 text-sm font-semibold transition ${
              voteState.userVote === "upvote"
                ? "border-white/60 text-slate-900 glass-tint-soft"
                : "border-white/60 text-slate-600 hover:text-slate-900 glass-panel"
            }`}
          >
            <ArrowBigUp className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={() => void handleVote("downvote")}
            className={`rounded-full border px-2 py-1 text-sm font-semibold transition ${
              voteState.userVote === "downvote"
                ? "border-white/60 text-slate-900 glass-tint-soft"
                : "border-white/60 text-slate-600 hover:text-slate-900 glass-panel"
            }`}
          >
            <ArrowBigDown className="h-4 w-4" />
          </button>
          <span className="text-xs font-semibold text-slate-500">{upvotePercentage}%</span>
        </div>
      )}
      <p className="mt-4 text-sm text-slate-700">{review.comment}</p>
    </div>
  );
}
