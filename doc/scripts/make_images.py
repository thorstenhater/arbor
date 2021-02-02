import copy
import svgwrite
import math
import inputs

tag_colors_colorscheme = ['white', '#ffc2c2', 'gray', '#c2caff', '#81c8aa']
tag_colors_bwscheme = ['lightgray']*len(tag_colors_colorscheme)

#
# ############################################
#

def translate(x, f, xshift):
    return (f*x[0]+xshift, -f*x[1])

def translate_all(points, f, xshift):
    return [translate(x, f, xshift) for x in points]

# Draw one or more morphologies, side by side.
# Each morphology can be drawn as segments or branches.
def morph_image(morphs, methods, filename, drawnumbers=True, colors=True, sc=20):
    assert(len(morphs)==len(methods))
    tag_colors = tag_colors_colorscheme if colors else tag_colors_bwscheme

    print('generating:', filename)
    dwg = svgwrite.Drawing(filename=filename, debug=True)

    # Width of lines and circle strokes.
    line_width=0.1*sc

    # Padding around image.
    fudge=1.5*sc

    linecolor='black'
    pointcolor='red'
    lines = dwg.add(dwg.g(id='lines',
                          stroke=linecolor,
                          fill='white',
                          stroke_width=line_width))
    points = dwg.add(dwg.g(id='points',
                           stroke=pointcolor,
                           fill=pointcolor,
                           stroke_width=line_width))
    numbers = dwg.add(dwg.g(id='numbers',
                             text_anchor='middle',
                             alignment_baseline='middle'))

    minx = math.inf
    miny = math.inf
    maxx = -math.inf
    maxy = -math.inf

    offset = 0

    bcolor = 'mediumslateblue'
    branchfillcolor = 'lightgray'

    nmorph = len(morphs)

    for l in range(nmorph):
        morph = morphs[l]
        method = methods[l]

        nbranches = len(morph)

        segid = 0
        for i in range(nbranches):
            branch = morph[i]

            lx, ux, ly, uy = branch.minmax()
            minx = min(minx,  sc*lx+offset)
            miny = min(miny,  sc*ly)
            maxx = max(maxx,  sc*ux+offset)
            maxy = max(maxy,  sc*uy)

            if method=='segments':
                for sec in branch.sections:
                    for seg in sec:
                        if seg.length>0.00001: # only draw nonzero length segments
                            line = translate_all(seg.corners(), sc, offset)
                            lines.add(dwg.polygon(points=line, fill=tag_colors[seg.tag]))

                            pos = translate(seg.location(0.5), sc, offset)
                            if drawnumbers:
                                points.add(dwg.circle(center=pos,
                                                    stroke='black',
                                                    r=sc*0.55,
                                                    fill='white'))
                                # The svg alignment_baseline attribute:
                                #   - works on Chrome/Chromium
                                #   - doesn't work on Firefox
                                # so for now we just shift the relative position by sc/3
                                label_pos = (pos[0], pos[1]+sc/3)
                                numbers.add(dwg.text(str(segid),
                                                    insert=label_pos,
                                                    stroke='black',
                                                    fill='black'))
                        segid += 1

            elif method=='branches':
                for line in branch.outline():
                    lines.add(dwg.polygon(points=translate_all(line, sc, offset),
                                          fill=branchfillcolor))

                if drawnumbers:
                    pos = translate(branch.location(0.5), sc, offset)
                    points.add(dwg.circle(center=pos,
                                        stroke=bcolor,
                                        r=sc*0.55,
                                        fill=bcolor))
                    # The svg alignment_baseline attribute:
                    #   - works on Chrome/Chromium
                    #   - doesn't work on Firefox
                    # so for now we just shift the relative position by sc/3
                    label_pos = (pos[0], pos[1]+sc/3)
                    numbers.add(dwg.text(str(i),
                                        insert=label_pos,
                                        stroke='white',
                                        fill='white'))
        offset = maxx - minx + sc


    # Find extent of image.
    minx -= fudge
    miny -= fudge
    maxx += fudge
    maxy += fudge
    width = maxx-minx
    height = maxy-miny
    dwg.viewbox(minx, -maxy, width, height)

    # Write the image to file.
    dwg.save()

# Generate an image that illustrates regions and locsets on a morphology.
#
# Can't handle morpholgies with gaps, where segemnts with a parent-child
# ordering don't have collocated distal-proximal locations respectively.
# Handling this case would make rendering regions more complex, but would
# not bee too hard to support.
def label_image(morphology, labels, filename, drawroot=True, sc=20):
    morph = morphology
    print('generating:', filename)
    dwg = svgwrite.Drawing(filename=filename, debug=True)

    # Width of lines and circle strokes.
    line_width=0.2*sc

    # Padding around image.
    fudge=1.5*sc

    linecolor='black'
    pointcolor='red'
    lines = dwg.add(dwg.g(id='lines',
                          stroke=linecolor,
                          fill='white',
                          stroke_width=line_width))
    points = dwg.add(dwg.g(id='points',
                           stroke=pointcolor,
                           fill=pointcolor,
                           stroke_width=line_width))

    minx = math.inf
    miny = math.inf
    maxx = -math.inf
    maxy = -math.inf

    offset = 0

    branchfillcolor = 'lightgray'

    nimage = len(labels)
    for l in range(nimage):
        lab = labels[l]

        nbranches = len(morph)

        # Draw the outline of the cell
        for i in range(nbranches):
            branch = morph[i]

            lx, ux, ly, uy = branch.minmax()
            minx = min(minx,  sc*lx+offset)
            miny = min(miny,  sc*ly)
            maxx = max(maxx,  sc*ux+offset)
            maxy = max(maxy,  sc*uy)

            for line in branch.outline():
                lines.add(dwg.polygon(points=translate_all(line, sc, offset),
                                      fill=branchfillcolor,
                                      stroke=branchfillcolor))

        # Draw the root
        root = translate(morph[0].location(0), sc, offset)
        if drawroot:
            points.add(dwg.circle(center=root, stroke='red', r=sc/2.5, fill='white'))

        if lab['type'] == 'locset':
            for loc in lab['value']:
                bid = loc[0]
                pos = loc[1]

                loc = translate(morph[bid].location(pos), sc, offset)
                points.add(dwg.circle(center=loc, stroke='black', r=sc/3, fill='white'))

        if lab['type'] == 'region':
            for cab in lab['value']:
                # skip zero length cables
                bid  = cab[0]
                ppos = cab[1]
                dpos = cab[2]

                # Don't draw zero-length cables
                # How should these be drawn: with a line or a circle?
                if ppos==dpos: continue

                for line in morph[bid].outline(ppos, dpos):
                    lines.add(dwg.polygon(points=translate_all(line, sc, offset),
                                          fill='black',
                                          stroke=branchfillcolor))

        offset = maxx - minx + sc

    # Find extent of image.
    minx -= fudge
    miny -= fudge
    maxx += fudge
    maxy += fudge
    width = maxx-minx
    height = maxy-miny
    dwg.viewbox(minx, -maxy, width, height)

    # Write the image to file.
    dwg.save()

def generate(path=''):

    morph_image([inputs.branch_morph2], ['segments'], path+'/term_segments.svg', False, False)
    morph_image([inputs.branch_morph2], ['branches'], path+'/term_branch.svg', False, False)
    label_image(inputs.branch_morph2, [inputs.reg_cable_0_28], path+'/term_cable.svg', False)

    morph_image([inputs.label_morph],    ['branches'], path+'/label_branch.svg')

    morph_image([inputs.label_morph],    ['segments'], path+'/label_seg.svg')
    morph_image([inputs.detached_morph], ['segments'], path+'/detached_seg.svg')
    morph_image([inputs.stacked_morph],  ['segments'], path+'/stacked_seg.svg')
    morph_image([inputs.swc_morph],      ['segments'], path+'/swc_morph.svg')

    morph_image([inputs.label_morph, inputs.label_morph], ['segments', 'branches'], path+'/label_morph.svg')
    morph_image([inputs.detached_morph, inputs.detached_morph], ['segments', 'branches'], path+'/detached_morph.svg')
    morph_image([inputs.stacked_morph, inputs.stacked_morph], ['segments', 'branches'], path+'/stacked_morph.svg')
    morph_image([inputs.sphere_morph, inputs.sphere_morph], ['segments', 'branches'], path+'/sphere_morph.svg')
    morph_image([inputs.branch_morph1, inputs.branch_morph1], ['segments', 'branches'], path+'/branch_morph1.svg')
    morph_image([inputs.branch_morph2, inputs.branch_morph2], ['segments', 'branches'], path+'/branch_morph2.svg')
    morph_image([inputs.branch_morph3, inputs.branch_morph3], ['segments', 'branches'], path+'/branch_morph3.svg')
    morph_image([inputs.branch_morph4, inputs.branch_morph4], ['segments', 'branches'], path+'/branch_morph4.svg')
    morph_image([inputs.yshaped_morph, inputs.yshaped_morph], ['segments', 'branches'], path+'/yshaped_morph.svg')
    morph_image([inputs.ysoma_morph1,  inputs.ysoma_morph1],  ['segments', 'branches'], path+'/ysoma_morph1.svg')
    morph_image([inputs.ysoma_morph2,  inputs.ysoma_morph2],  ['segments', 'branches'], path+'/ysoma_morph2.svg')
    morph_image([inputs.ysoma_morph3,  inputs.ysoma_morph3],  ['segments', 'branches'], path+'/ysoma_morph3.svg')

    ####################### locsets

    label_image(inputs.label_morph, [inputs.ls_term, inputs.ls_rand_dend], path+'/locset_label_examples.svg')
    label_image(inputs.label_morph, [inputs.reg_dend, inputs.reg_radlt5], path+'/region_label_examples.svg')
    label_image(inputs.label_morph, [inputs.ls_root], path+'/root_label.svg')
    label_image(inputs.label_morph, [inputs.ls_term], path+'/term_label.svg')
    label_image(inputs.label_morph, [inputs.ls_loc15], path+'/location_label.svg')
    label_image(inputs.label_morph, [inputs.reg_rad36, inputs.ls_distal], path+'/distal_label.svg')
    label_image(inputs.label_morph, [inputs.reg_rad36, inputs.ls_proximal], path+'/proximal_label.svg')
    label_image(inputs.label_morph, [inputs.ls_uniform0, inputs.ls_uniform1], path+'/uniform_label.svg')
    label_image(inputs.label_morph, [inputs.ls_branchmid], path+'/on_branches_label.svg')
    label_image(inputs.label_morph, [inputs.ls_term, inputs.reg_tag3, inputs.ls_restrict], path+'/restrict_label.svg')

    ####################### regions

    label_image(inputs.label_morph, [inputs.reg_empty, inputs.reg_all], path+'/nil_all_label.svg')
    label_image(inputs.label_morph, [inputs.reg_tag1, inputs.reg_tag2, inputs.reg_tag3], path+'/tag_label.svg')
    label_image(inputs.label_morph, [inputs.reg_tag1, inputs.reg_tag3], path+'/tag_label.svg')
    label_image(inputs.label_morph, [inputs.reg_branch0, inputs.reg_branch3], path+'/branch_label.svg')
    label_image(inputs.label_morph, [inputs.reg_segment0, inputs.reg_segment3], path+'/segment_label.svg')
    label_image(inputs.label_morph, [inputs.reg_cable_1_01, inputs.reg_cable_1_31, inputs.reg_cable_1_37], path+'/cable_label.svg')
    label_image(inputs.label_morph, [inputs.ls_proxint_in, inputs.reg_proxint],    path+'/proxint_label.svg')
    label_image(inputs.label_morph, [inputs.ls_proxint_in, inputs.reg_proxintinf], path+'/proxintinf_label.svg')
    label_image(inputs.label_morph, [inputs.ls_distint_in, inputs.reg_distint],    path+'/distint_label.svg')
    label_image(inputs.label_morph, [inputs.ls_distint_in, inputs.reg_distintinf], path+'/distintinf_label.svg')
    label_image(inputs.label_morph, [inputs.reg_lhs, inputs.reg_rhs, inputs.reg_or],  path+'/union_label.svg')
    label_image(inputs.label_morph, [inputs.reg_lhs, inputs.reg_rhs, inputs.reg_and], path+'/intersect_label.svg')
    label_image(inputs.label_morph, [inputs.reg_radlt5],  path+'/radiuslt_label.svg')
    label_image(inputs.label_morph, [inputs.reg_radle5],  path+'/radiusle_label.svg')
    label_image(inputs.label_morph, [inputs.reg_radgt5],  path+'/radiusgt_label.svg')
    label_image(inputs.label_morph, [inputs.reg_radge5],  path+'/radiusge_label.svg')

    ####################### Tutorial examples

    morph_image([inputs.tutorial_morph], ['segments'], path+'/tutorial_morph.svg')

    ####################### locsets

    label_image(inputs.tutorial_morph, [inputs.tut_ls_root, inputs.tut_ls_terminal], path+'/tutorial_root_term.svg')
    label_image(inputs.tutorial_morph, [inputs.tut_ls_custom_terminal, inputs.tut_ls_axon_terminal], path+'/tutorial_custom_axon_term.svg')


    ####################### regions
    label_image(inputs.tutorial_morph, [inputs.tut_reg_soma, inputs.tut_reg_axon, inputs.tut_reg_dend, inputs.tut_reg_last], path+'/tutorial_tag.svg')
    label_image(inputs.tutorial_morph, [inputs.tut_reg_all, inputs.tut_reg_rad_gt], path+'/tutorial_all_gt.svg')
    label_image(inputs.tutorial_morph, [inputs.tut_reg_custom], path+'/tutorial_custom.svg')

if __name__ == '__main__':
    generate('.')
