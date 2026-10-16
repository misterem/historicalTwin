import type { Metadata, Viewport } from "next";
import { EB_Garamond, Playfair_Display, Playfair_Display_SC } from "next/font/google";
import "./globals.css";

const display = Playfair_Display({
  variable: "--font-playfair",
  subsets: ["latin"],
  weight: ["500", "600", "800"],
  style: ["normal", "italic"],
});

const caps = Playfair_Display_SC({
  variable: "--font-playfair-sc",
  subsets: ["latin"],
  weight: ["400", "700"],
});

const body = EB_Garamond({
  variable: "--font-garamond",
  subsets: ["latin"],
  style: ["normal", "italic"],
});

export const metadata: Metadata = {
  title: "Historical Twin",
  description: "Take a selfie and meet the portrait painting that looks the most like you.",
};

export const viewport: Viewport = {
  themeColor: "#2e0b0e",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${display.variable} ${caps.variable} ${body.variable}`}>
      <body>{children}</body>
    </html>
  );
}
