# CSS Overrides for PaperMod Theme

This directory contains custom CSS overrides for the PaperMod theme. Any CSS files placed here will be automatically loaded after the theme's core CSS, ensuring your custom styles take precedence.

## How It Works

1. Files in this directory are automatically picked up by Hugo's asset pipeline
2. They are loaded after the theme's core CSS files
3. This ensures your custom styles can override theme defaults

## CSS Loading Order

1. Theme Core CSS (loaded first):
   - theme-vars.css (CSS variables)
   - reset.css
   - common/*.css
   - chroma-styles.css
   - zmedia.css

2. Extended CSS (loaded last):
   - All files in this directory (including custom-override.css)
   - These styles override any previous styles

## Usage

1. To override theme variables:
   ```css
   :root {
       --main-width: 1200px;  /* Example override */
   }
   ```

2. To add custom styles:
   ```css
   .your-custom-class {
       /* Your styles here */
   }
   ```

## Best Practices

1. Use descriptive file names (e.g., `custom-override.css`)
2. Add clear comments explaining your overrides
3. Group related styles together
4. Use the theme's existing variables when possible

## Files

- `custom-override.css`: Main override file for theme-wide customizations 