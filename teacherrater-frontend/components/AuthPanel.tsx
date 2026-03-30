"use client";

import { useEffect, useLayoutEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { X } from "lucide-react";

import { apiFetch, ensureCsrfToken } from "@/lib/api";
import { loadToken, saveToken, saveUser } from "@/lib/auth";
import { useSchool } from "@/components/SchoolProvider";
import { UserProfile, SchoolKey } from "@/lib/types";
import { loadSchool, persistSchool, schoolThemes } from "@/lib/school";

const MODE_OPTIONS = ["login", "signup"] as const;
const SCHOOL_OPTIONS: SchoolKey[] = ["sky", "iss", "lib"];

export function AuthPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { school, setSchool } = useSchool();
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [identifier, setIdentifier] = useState("");
  const [password, setPassword] = useState("");
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [message, setMessage] = useState<string | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [lastLoginAttempt, setLastLoginAttempt] = useState<number | null>(null);
  const [lastSignupAttempt, setLastSignupAttempt] = useState<number | null>(null);
  const [signupSchool, setSignupSchool] = useState<SchoolKey>(school);
  const [isClosing, setIsClosing] = useState(false);
  const [isAnimatingIn, setIsAnimatingIn] = useState(false);
  const LOGIN_COOLDOWN_MS = 10_000;
  const SIGNUP_COOLDOWN_MS = 10_000;
  const USERNAME_PATTERN = /^[A-Za-z0-9 ,.?!]+$/;

  useEffect(() => {
    const updateToken = () => {
      const storedToken = loadToken();
      setToken(storedToken);
      if (!storedToken) {
        setMessage(null);
      }
    };
    updateToken();
    window.addEventListener("auth-updated", updateToken);
    return () => window.removeEventListener("auth-updated", updateToken);
  }, []);

  useEffect(() => {
    if (!open) {
      setMessage(null);
      setPassword("");
      setMode("login");
      setSignupSchool(school);
    }
  }, [open]);

  useLayoutEffect(() => {
    if (open) {
      setIsClosing(false);
      setIsAnimatingIn(true);
      const animationFrame = window.requestAnimationFrame(() => {
        setIsAnimatingIn(false);
      });
      return () => window.cancelAnimationFrame(animationFrame);
    }
    setIsAnimatingIn(false);
  }, [open]);

  useEffect(() => {
    if (token) {
      setMessage(null);
    }
  }, [token]);

  useEffect(() => {
    if (!open || mode !== "signup") {
      return;
    }
    const previousTheme = document.documentElement.getAttribute("data-theme");
    document.documentElement.setAttribute("data-theme", schoolThemes[signupSchool].theme);
    return () => {
      if (previousTheme) {
        document.documentElement.setAttribute("data-theme", previousTheme);
      } else {
        document.documentElement.removeAttribute("data-theme");
      }
    };
  }, [open, mode, signupSchool]);

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    setMessage(null);
    const now = Date.now();
    if (mode === "login" && lastLoginAttempt && now - lastLoginAttempt < LOGIN_COOLDOWN_MS) {
      setMessage("Please wait before trying to log in again.");
      return;
    }
    if (mode === "signup" && lastSignupAttempt && now - lastSignupAttempt < SIGNUP_COOLDOWN_MS) {
      setMessage("Please wait before trying to sign up again.");
      return;
    }
    if (mode === "signup" && !USERNAME_PATTERN.test(identifier)) {
      setMessage("Username contains unsupported characters.");
      return;
    }
    setSubmitting(true);
    const activeSchool = mode === "signup" ? signupSchool : school;
    try {
      await ensureCsrfToken(activeSchool);
      if (mode === "signup") {
        setLastSignupAttempt(now);
        await apiFetch("/auth/signup", {
          method: "POST",
          school: signupSchool,
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username: identifier, email, password, honeypot: "" }),
        });
        persistSchool(signupSchool);
        setMode("login");
        setMessage("Signup complete. Please log in.");
        return;
      }
      setLastLoginAttempt(now);
      const response = await apiFetch<{ access_token: string }>("/auth/login", {
        method: "POST",
        school,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ identifier, password, honeypot: "" }),
      });
      saveToken(response.access_token);
      setToken(response.access_token);
      const profile = await apiFetch<UserProfile>("/auth/me", {
        school,
        token: response.access_token,
      });
      saveUser(profile);
      setSchool(loadSchool());
      if (profile.is_moderator) {
        router.push("/moderators");
      }
      setMessage("Logged in successfully.");
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Unable to authenticate.");
    } finally {
      setSubmitting(false);
    }
  }
  function handleIdentifierChange(event: React.ChangeEvent<HTMLInputElement>) {
    setIdentifier(event.target.value);
    setMessage(null);
  }

  function handleEmailChange(event: React.ChangeEvent<HTMLInputElement>) {
    setEmail(event.target.value);
    setMessage(null);
  }

  function handlePasswordChange(event: React.ChangeEvent<HTMLInputElement>) {
    setPassword(event.target.value);
    setMessage(null);
  }

  function handleModeChange(nextMode: "login" | "signup") {
    if (nextMode === "signup") {
      if (identifier.includes("@")) {
        setEmail(identifier);
        setIdentifier("");
      } else {
        setEmail("");
      }
    } else if (!identifier && email) {
      setIdentifier(email);
      setEmail("");
    }
    setMode(nextMode);
    setMessage(null);
  }

  function handleSchoolChange(nextSchool: SchoolKey) {
    setSignupSchool(nextSchool);
  }

  if (token || (!open && !isClosing)) {
    return null;
  }

  const isVisible = open && !isClosing;

  function handleClose() {
    setIsClosing(true);
    window.setTimeout(() => {
      onClose();
      setIsClosing(false);
    }, 280);
  }

  return (
    <div
      className={`fixed inset-0 z-40 flex items-center justify-center px-4 backdrop-blur-sm transition-opacity duration-[280ms] ${
        isVisible ? "bg-slate-900/15 opacity-100" : "bg-slate-900/0 opacity-0"
      }`}
      onClick={handleClose}
    >
      <div
        className={`w-full max-w-sm rounded-2xl p-6 glass-panel-strong transition-[opacity,transform] duration-[280ms] ${
          isVisible && !isAnimatingIn
            ? "translate-y-0 scale-100 opacity-100"
            : "translate-y-4 scale-95 opacity-0"
        }`}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-slate-900">Account</h3>
          <button
            type="button"
            onClick={handleClose}
            className="rounded-full p-2 text-slate-500 transition hover:bg-white/40 hover:text-slate-700"
            aria-label="Close login panel"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <form onSubmit={handleSubmit} className="mt-4 flex flex-col gap-3">
          <div className="relative grid grid-cols-2 rounded-full p-1 glass-panel">
            <span
              className="absolute left-1 top-1 h-[calc(100%-0.5rem)] w-[calc(50%-0.25rem)] rounded-full border border-white/40 glass-tint-medium transition-transform duration-300"
              style={{
                transform: `translateX(${MODE_OPTIONS.indexOf(mode) * 100}%)`,
              }}
            />
            {MODE_OPTIONS.map((option) => (
              <button
                key={option}
                type="button"
                onClick={() => handleModeChange(option)}
                className={`relative z-10 rounded-full px-3 py-1 text-sm font-semibold transition-colors duration-300 ${
                  mode === option ? "text-white" : "text-slate-700 hover:text-slate-900"
                }`}
              >
                {option === "login" ? "Login" : "Signup"}
              </button>
            ))}
          </div>
          <div
            className={`overflow-hidden transition-[max-height,opacity,transform,margin] duration-300 ease-out ${
              mode === "signup" ? "max-h-16 opacity-100 translate-y-0 my-0" : "max-h-0 opacity-0 -translate-y-1 -my-1.5"
            }`}
          >
            <div className="relative grid grid-cols-3 rounded-full p-1 text-xs font-semibold glass-panel">
              <span
                className="absolute left-1 top-1 h-[calc(100%-0.5rem)] w-[calc(33.333%-0.25rem)] rounded-full border border-white/40 glass-tint-medium transition-transform duration-300"
                style={{
                  transform: `translateX(${SCHOOL_OPTIONS.indexOf(signupSchool) * 100}%)`,
                }}
              />
              {SCHOOL_OPTIONS.map((option) => (
                <button
                  key={option}
                  type="button"
                  onClick={() => handleSchoolChange(option)}
                  className={`relative z-10 rounded-full px-3 py-1 transition-colors duration-300 ${
                    signupSchool === option ? "text-white" : "text-slate-700 hover:text-slate-900"
                  }`}
                >
                  {option}
                </button>
              ))}
            </div>
          </div>
          <input
            type="text"
            placeholder={mode === "login" ? "Username or email" : "Username"}
            value={identifier}
            onChange={handleIdentifierChange}
            maxLength={mode === "signup" ? 32 : undefined}
            className="w-full rounded-lg px-3 py-2 glass-input"
            required
          />
          <div
            className={`overflow-hidden transition-[max-height,opacity,transform,margin] duration-300 ease-out ${
              mode === "signup"
                ? "max-h-24 opacity-100 translate-y-0 my-0"
                : "max-h-0 opacity-0 -translate-y-1 -my-1.5"
            }`}
          >
            <input
              type="email"
              placeholder="Email"
              value={email}
              onChange={handleEmailChange}
              className="w-full rounded-lg px-3 py-2 glass-input"
              required={mode === "signup"}
              tabIndex={mode === "signup" ? 0 : -1}
            />
          </div>
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={handlePasswordChange}
            maxLength={64}
            minLength={8}
            className="w-full rounded-lg px-3 py-2 glass-input"
            required
          />
          <button
            type="submit"
            disabled={submitting}
            className="w-full rounded-full border border-white/40 px-4 py-2 text-sm font-semibold text-white shadow-md transition glass-tint-medium backdrop-blur-xl disabled:opacity-60"
          >
            {mode === "login" ? "Log in" : "Sign up"}
          </button>
        </form>
        {message && <p className="mt-3 text-sm text-slate-600">{message}</p>}
      </div>
    </div>
  );
}
