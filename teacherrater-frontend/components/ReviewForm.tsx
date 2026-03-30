"use client";

import { useState } from "react";

import { apiFetch, ensureCsrfToken } from "@/lib/api";
import { loadToken } from "@/lib/auth";
import { useSchool } from "@/components/SchoolProvider";

export function ReviewForm({ teacherId, onSubmitted }: { teacherId: number; onSubmitted: () => void }) {
  const { school } = useSchool();
  const [rating, setRating] = useState(5);
  const [difficulty, setDifficulty] = useState(3);
  const [clarity, setClarity] = useState(4);
  const [comment, setComment] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const allowedPattern = /^[A-Za-z0-9 ,.?!]+$/;
  const COMMENT_MAX_LENGTH = 200;

  const token = loadToken();

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!token) {
      setError("Please log in to post a review.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      if (!allowedPattern.test(comment)) {
        setError("Comment contains unsupported characters.");
        setSubmitting(false);
        return;
      }
      await ensureCsrfToken(school);
      await apiFetch(`/teachers/${teacherId}/reviews`, {
        method: "POST",
        school,
        token,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rating, difficulty, clarity, comment, honeypot: "" }),
      });
      setComment("");
      onSubmitted();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to submit review.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="rounded-2xl p-6 glass-panel">
      <h3 className="text-lg font-semibold text-slate-900">Leave a review</h3>
      <div className="mt-4 grid gap-4 md:grid-cols-3">
        <label className="text-sm text-slate-600">
          Overall rating
          <input
            type="number"
            min={1}
            max={5}
            value={rating}
            onChange={(event) => setRating(Number(event.target.value))}
            className="mt-2 w-full rounded-lg px-3 py-2 glass-input"
          />
        </label>
        <label className="text-sm text-slate-600">
          Difficulty
          <input
            type="number"
            min={1}
            max={5}
            value={difficulty}
            onChange={(event) => setDifficulty(Number(event.target.value))}
            className="mt-2 w-full rounded-lg px-3 py-2 glass-input"
          />
        </label>
        <label className="text-sm text-slate-600">
          Clarity
          <input
            type="number"
            min={1}
            max={5}
            value={clarity}
            onChange={(event) => setClarity(Number(event.target.value))}
            className="mt-2 w-full rounded-lg px-3 py-2 glass-input"
          />
        </label>
      </div>
      <label className="mt-4 block text-sm text-slate-600">
        Comment
        <textarea
          value={comment}
          onChange={(event) => setComment(event.target.value)}
          rows={4}
          className="mt-2 w-full rounded-lg px-3 py-2 glass-input"
          maxLength={COMMENT_MAX_LENGTH}
          minLength={3}
          required
        />
      </label>
      <p className="text-xs text-slate-500">{COMMENT_MAX_LENGTH} characters max</p>
      {error && <p className="mt-2 text-sm text-red-600">{error}</p>}
      <button
        type="submit"
        disabled={submitting}
        className="mt-4 rounded-full border border-white/40 px-4 py-2 text-sm font-semibold text-white shadow-md transition glass-tint backdrop-blur-xl disabled:opacity-60"
      >
        {submitting ? "Submitting..." : "Submit review"}
      </button>
    </form>
  );
}
