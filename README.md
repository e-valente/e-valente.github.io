# Emanuel Valente's Blog

Source code for [blog.e-valente.io](https://blog.e-valente.io/), built with
[Hugo](https://gohugo.io/) and the [PaperMod](https://github.com/adityatelange/hugo-PaperMod)
theme. The site is published to GitHub Pages when changes are pushed to `main`.

## Requirements

- Git
- **Hugo Extended 0.167.0 or newer**. The GitHub Actions deployment currently
  builds with Hugo Extended 0.167.0.

### Install Hugo Extended

On macOS, install it with Homebrew:

```sh
brew install hugo
```

On Ubuntu or another Linux distribution with Snap support:

```sh
sudo snap install hugo --channel=extended
```

For other Linux distributions, download the **extended** package from the
[official Hugo releases](https://github.com/gohugoio/hugo/releases) and add
the `hugo` binary to your `PATH`.

Confirm the installed version:

```sh
hugo version
```

## Run locally

Clone the repository and fetch the theme submodule in one step:

```sh
git clone --recurse-submodules git@github.com:e-valente/e-valente.github.io.git
cd e-valente.github.io
```

If the repository was cloned without submodules, initialize them before
running Hugo:

```sh
git submodule update --init --recursive
```

The PaperMod theme is tracked as a Git submodule in `themes/PaperMod`; no Ruby,
Bundler, or Jekyll dependency installation is required.

Start the development server, including draft posts:

```sh
hugo server -D
```

Open the local address displayed by Hugo (normally `http://localhost:1313/`).

To perform the same optimized build used in CI:

```sh
hugo --gc --minify
```

Generated files are written to `public/` and are not committed.

## Create a blog post

1. Create a Markdown file under `content/posts/`, using the naming convention
   `YYYY-MM-DD-descriptive-slug.md`:

   ```sh
   hugo new content content/posts/2026-11-02-my-new-post.md
   ```

   You can also create the file manually if you prefer.

2. Use this front matter as a starting point. Keep the date in ISO 8601 format
   and include its timezone.

   ```md
   ---
   title: "My New Post"
   date: 2026-11-02T10:00:00-03:00
   tags: ["security", "research"]
   ShowToc: true
   draft: true
   ---

   Start writing here.
   ```

   - Use short, lowercase, hyphenated slugs in filenames.
   - Add only relevant tags; they are used by the site's tag index.
   - Set `draft: false` (or remove it) before publishing.
   - Add `aliases` when replacing or moving an existing URL, for example:

     ```yaml
     aliases:
       - "/old-post-url/"
     ```

3. Write the post in Markdown. Store local images in `static/` or `assets/`
   and reference them with site-relative URLs, for example
   `/assets/img/example.png`. Use descriptive alt text for images.

4. Review the result locally with `hugo server -D`, then validate it:

   ```sh
   hugo --gc --minify
   ```

5. Commit and push the post to `main`. The workflow in
   `.github/workflows/deploy-hugo.yml` builds the site and deploys it to GitHub
   Pages.

## Post ideas

- Amazonlinux2023 - Add security patch repo
- Machine Learning Security Blog (semcomp)
- MQTT Security Labs
- DoH Security
- MLSec - Different Gradients and its implications on attacks
