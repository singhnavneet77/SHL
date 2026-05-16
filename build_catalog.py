"""
Build SHL catalog from all fetched page data.
Parse the markdown table format from the pages we've fetched.
"""
import re, json

# All page data collected from web_fetch calls
# We extract ONLY the Individual Test Solutions table rows

all_pages_data = """
PAGE_1_START=0:
| Individual Test Solutions | Remote Testing | Adaptive/IRT | Test Type |
| [Global Skills Development Report](https://www.shl.com/products/product-catalog/view/global-skills-development-report/) |  |  | A E B C D P |
| [.NET Framework 4.5](https://www.shl.com/products/product-catalog/view/net-framework-4-5/) |  |  | K |
| [.NET MVC (New)](https://www.shl.com/products/product-catalog/view/net-mvc-new/) |  |  | K |
| [.NET MVVM (New)](https://www.shl.com/products/product-catalog/view/net-mvvm-new/) |  |  | K |
| [.NET WCF (New)](https://www.shl.com/products/product-catalog/view/net-wcf-new/) |  |  | K |
| [.NET WPF (New)](https://www.shl.com/products/product-catalog/view/net-wpf-new/) |  |  | K |
| [.NET XAML (New)](https://www.shl.com/products/product-catalog/view/net-xaml-new/) |  |  | K |
| [Accounts Payable (New)](https://www.shl.com/products/product-catalog/view/accounts-payable-new/) |  |  | K |
| [Accounts Payable Simulation (New)](https://www.shl.com/products/product-catalog/view/accounts-payable-simulation-new/) |  |  | S |
| [Accounts Receivable (New)](https://www.shl.com/products/product-catalog/view/accounts-receivable-new/) |  |  | K |
| [Accounts Receivable Simulation (New)](https://www.shl.com/products/product-catalog/view/accounts-receivable-simulation-new/) |  |  | S |
| [ADO.NET (New)](https://www.shl.com/products/product-catalog/view/ado-net-new/) |  |  | K |

PAGE_2_START=12:
| [Adobe Experience Manager (New)](https://www.shl.com/products/product-catalog/view/adobe-experience-manager-new/) |  |  | K |
| [Adobe Photoshop CC](https://www.shl.com/products/product-catalog/view/adobe-photoshop-cc/) |  |  | K |
| [Aeronautical Engineering (New)](https://www.shl.com/products/product-catalog/view/aeronautical-engineering-new/) |  |  | K |
| [Aerospace Engineering (New)](https://www.shl.com/products/product-catalog/view/aerospace-engineering-new/) |  |  | K |
| [Agile Software Development](https://www.shl.com/products/product-catalog/view/agile-software-development/) |  |  | K |
| [Agile Testing (New)](https://www.shl.com/products/product-catalog/view/agile-testing-new/) |  |  | K |
| [AI Skills](https://www.shl.com/products/product-catalog/view/ai-skills/) |  |  | P |
| [Amazon Web Services (AWS) Development (New)](https://www.shl.com/products/product-catalog/view/amazon-web-services-aws-development-new/) |  |  | K |
| [Android Development (New)](https://www.shl.com/products/product-catalog/view/android-development-new/) |  |  | K |
| [Angular 6 (New)](https://www.shl.com/products/product-catalog/view/angular-6-new/) |  |  | K |
| [AngularJS (New)](https://www.shl.com/products/product-catalog/view/angularjs-new/) |  |  | K |
| [Apache Hadoop (New)](https://www.shl.com/products/product-catalog/view/apache-hadoop-new/) |  |  | K |
"""

def parse_products(text):
    products = {}
    pattern = r'\[([^\]]+)\]\((https://www\.shl\.com/products/product-catalog/view/[^)]+)\)\s*\|[^|]*\|[^|]*\|\s*([A-Z ,]+)\s*\|?'
    for m in re.finditer(pattern, text):
        name = m.group(1).strip()
        url = m.group(2).strip()
        test_type = m.group(3).strip()
        products[url] = {"name": name, "url": url, "test_type": test_type}
    return products

parsed = parse_products(all_pages_data)
print(f"Parsed {len(parsed)} products from collected page data")
for u, p in parsed.items():
    print(f"  {p['name']} | {p['test_type']}")
