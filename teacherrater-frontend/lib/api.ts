import { SchoolKey } from "./types";

const DEFAULT_API_URL = "http://localhost:3002";
const rawApiUrl = process.env.NEXT_PUBLIC_API_URL;
const normalizedApiUrl = rawApiUrl
  ?.replace("http://localhost:8000", DEFAULT_API_URL)
  .replace("http://127.0.0.1:8000", DEFAULT_API_URL);
const API_URL = normalizedApiUrl || DEFAULT_API_URL;
let csrfTokenCache: string | undefined;

function getCsrfToken(): string | undefined {
  return csrfTokenCache;
}

export async function ensureCsrfToken(school: SchoolKey): Promise<void> {
  const response = await fetch(`${API_URL}/auth/csrf`, {
    credentials: "include",
    headers: { "X-School": school },
  });
  if (response.ok) {
    const payload = (await response.json().catch(() => null)) as { csrf_token?: string } | null;
    if (payload?.csrf_token) {
      csrfTokenCache = payload.csrf_token;
    }
  }
}

export async function apiFetch<T>(
  path: string,
  options: RequestInit & { school: SchoolKey; token?: string } = { school: "sky" }
): Promise<T> {
  const headers = new Headers(options.headers || {});
  headers.set("X-School", options.school);
  if (options.token) {
    headers.set("Authorization", `Bearer ${options.token}`);
  }
  if (options.method && options.method !== "GET") {
    const csrfToken = getCsrfToken();
    if (csrfToken) {
      headers.set("X-CSRF-Token", csrfToken);
    }
  }

  const response = await fetch(`${API_URL}${path}`, {
    ...options,
    headers,
    credentials: "include",
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Request failed" }));
    let message = "Request failed";
    if (error?.detail) {
      if (typeof error.detail === "string") {
        message = error.detail;
      } else if (Array.isArray(error.detail)) {
        message = error.detail.map((item) => item?.msg || JSON.stringify(item)).join(", ");
      } else {
        message = JSON.stringify(error.detail);
      }
    }
    throw new Error(message);
  }
  if (response.status === 204) {
    return null as T;
  }
  return (await response.json()) as T;
}
