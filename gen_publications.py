import json
from collections import defaultdict

# Function to convert JSON to Hugo format
def convert_json_to_hugo(publications):
    # Organize publications by year
    publications_by_year = defaultdict(list)
    for publication in publications:
        year = publication["date"].split("/")[1]
        if publication["venue"]:
            venue_year = publication["venue"].split(" ")[-1]
            try:
                if int(venue_year) > 2010:
                    year = venue_year
            except ValueError as e:
                print(f"The publication {publication} has year {venue_year} wrong")
                raise e
        publications_by_year[year].append(publication)
    # Sort years in descending order
    sorted_years = sorted(publications_by_year.keys(), reverse=True)
    # Format publications for Hugo
    hugo_output = ""
    for year in sorted_years:
        hugo_output += f"### {year}\n\n"
        for publication in publications_by_year[year]:
            title = publication["title"]
            venue = publication.get("venue", "")
            paper_link = publication.get("pdf", "")
            code_link = publication.get("code", "")
            authors = publication["authors"]
            # Assuming 'data-topic' needs to be manually adjusted or derived from 'tag'
            data_topic = publication["tag"] # This is a placeholder, adjust as needed
            award = publication["award"]
            project = publication["project"]

            hugo_output += (f"{{{{< publication title=\"{title}\" venue=\"{venue}\" "
                            f"paperLink=\"{paper_link}\" codeLink=\"{code_link}\" "
                            f"award=\"{award}\" project=\"{project}\" data-topic=\"{data_topic}\" >}}}}\n")
            hugo_output += f"{authors}\n"
            hugo_output += "{{< /publication >}}\n\n"
        hugo_output += "&emsp;\n\n"

    return hugo_output

# Load JSON data from file
with open('publications.json', 'r') as file:
    publications_json = json.load(file)

# Convert and print the Hugo formatted string
hugo_format = convert_json_to_hugo(publications_json)
print(hugo_format)
