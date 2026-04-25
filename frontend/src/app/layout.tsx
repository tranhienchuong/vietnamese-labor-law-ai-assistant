import type { Metadata } from "next"
import Script from "next/script"
import { GeistSans } from "geist/font/sans"
import "./globals.css"
import { APP_NAME, APP_TAGLINE } from "@/lib/constants"

export const metadata: Metadata = {
  title: APP_NAME,
  description: APP_TAGLINE
}

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="vi" suppressHydrationWarning>
      <body className={GeistSans.className} suppressHydrationWarning>
        <Script
          id="strip-extension-hydration-attributes"
          strategy="beforeInteractive"
          dangerouslySetInnerHTML={{
            __html: `
              (function () {
                function shouldRemoveAttribute(name) {
                  return name.indexOf('bis_') === 0 || name.indexOf('__processed_') === 0;
                }

                function cleanElement(element) {
                  if (!element || !element.attributes) return;
                  Array.prototype.slice.call(element.attributes).forEach(function (attribute) {
                    if (shouldRemoveAttribute(attribute.name)) {
                      element.removeAttribute(attribute.name);
                    }
                  });
                }

                function cleanTree(root) {
                  cleanElement(root);
                  if (!root || !root.querySelectorAll) return;
                  root.querySelectorAll('*').forEach(cleanElement);
                }

                function observe() {
                  if (!document.documentElement || !window.MutationObserver) return;
                  var observer = new MutationObserver(function (mutations) {
                    mutations.forEach(function (mutation) {
                      if (mutation.type === 'attributes') {
                        cleanElement(mutation.target);
                        return;
                      }

                      mutation.addedNodes.forEach(function (node) {
                        if (node.nodeType === 1) {
                          cleanTree(node);
                        }
                      });
                    });
                  });
                  observer.observe(document.documentElement, {
                    attributes: true,
                    childList: true,
                    subtree: true
                  });
                }

                if (document.readyState === 'loading') {
                  document.addEventListener('DOMContentLoaded', function () {
                    cleanTree(document.documentElement);
                  }, { once: true });
                } else {
                  cleanTree(document.documentElement);
                }
                observe();
              })();
            `
          }}
        />
        {children}
      </body>
    </html>
  )
}
