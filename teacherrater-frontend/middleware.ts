import { NextRequest, NextResponse } from "next/server";

export function middleware(request: NextRequest) {
  const isModeratorOnly = process.env.MODERATOR_ONLY === "true";
  if (!isModeratorOnly) {
    return NextResponse.next();
  }

  const { pathname } = request.nextUrl;
  if (pathname === "/moderators") {
    return NextResponse.next();
  }

  const url = request.nextUrl.clone();
  url.pathname = "/moderators";
  return NextResponse.rewrite(url);
}

export const config = {
  matcher: ["/((?!_next|api|favicon.ico).*)"],
};
