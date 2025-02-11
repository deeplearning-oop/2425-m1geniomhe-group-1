#!/bin/bash

# Function to increment the version
increment_version() {
    local version="$1"
    local part="$2"

    # Split version into components
    IFS='.' read -r major minor patch <<< "$version"

    # Increment the appropriate part
    case $part in
        x)
            major=$((major + 1))
            minor=0
            patch=0
            ;;
        y)
            minor=$((minor + 1))
            patch=0
            ;;
        z)
            patch=$((patch + 1))
            ;;
        *)
            echo "Invalid part: $part. Use 'x', 'y', or 'z'."
            exit 1
            ;;
    esac

    # Return the new version
    echo "$major.$minor.$patch"
}

# Check if VERSION file exists
if [[ ! -f VERSION ]]; then
    echo "VERSION file not found."
    exit 1
fi

# Read the current version
current_version=$(<VERSION)

# Ask user for the part to increment
read -p "Current version is $current_version. Increment which part? (x/y/z): " part
if [[ "$part" != "x" && "$part" != "y" && "$part" != "z" ]]; then
    echo "Invalid input: $part. Please use 'x', 'y', or 'z'."
    exit 1
fi

# Increment the version
new_version=$(increment_version "$current_version" "$part")

# Ask user for changelog entries
echo "Describe the changes for version $new_version. Enter each change as a bullet point."
echo "Press Enter on an empty line when done."
changelog_entries=()
while IFS= read -r line; do
    [[ -z "$line" ]] && break
    changelog_entries+=("$line")
done

if [[ ${#changelog_entries[@]} -eq 0 ]]; then
    echo "No changes provided. Exiting."
    exit 1
fi

# Update VERSION file
if ! echo "$new_version" > VERSION; then
    echo "Failed to update VERSION file."
    exit 1
fi
echo "Updated VERSION to $new_version."

# # Ensure the docs directory exists
# if [[ ! -d docs ]]; then
#     mkdir -p docs
#     echo "Created docs directory."
# fi

# Add entry to dev/changelog.md
changelog_file="dev/changelog.md"
if [[ ! -f $changelog_file ]]; then
    echo "# Changelog" > "$changelog_file"
fi

{
    echo "## [$new_version] - $(date +'%Y-%m-%d')"
    for entry in "${changelog_entries[@]}"; do
        echo "- $entry"
    done
    echo
} >> "$changelog_file" || {
    echo "Failed to update $changelog_file."
    exit 1
}

echo "Added changes to $changelog_file."
echo "Version updated successfully from $current_version to $new_version."
