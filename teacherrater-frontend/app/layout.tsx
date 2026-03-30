import "./globals.css";

import { Footer } from "@/components/Footer";
import { Header } from "@/components/Header";
import { SchoolProvider } from "@/components/SchoolProvider";

export const metadata = {
  title: "IDS Teachers",
  description: "Rate and review teachers across multiple schools.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="flex min-h-screen flex-col lowercase">
        <SchoolProvider>
          <Header />
          <main className="mx-auto w-full max-w-5xl flex-1 px-4 pb-8 pt-4">
            {children}
          </main>
          <Footer />
        </SchoolProvider>
      </body>
    </html>
  );
}
